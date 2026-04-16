from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from multi_doc_chat.config.settings import Settings, get_settings
from multi_doc_chat.exception.exceptions import QueryError, VectorStoreNotFoundError
from multi_doc_chat.logger.logger import get_logger
from multi_doc_chat.model.rag_model import CoTAnswer
from multi_doc_chat.prompts.templates import (
    build_contextualize_prompt,
    build_conversational_qa_prompt,
    build_cot_conversational_qa_prompt,
    build_qa_prompt,
)

logger = get_logger(__name__)


def _format_docs(docs: List[Document]) -> str:
    """Join retrieved Document page_content with double newlines.

    Args:
        docs: List of retrieved LangChain Documents.

    Returns:
        str: Concatenated context string ready to inject into the prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)


class RAGQueryEngine:
    """Loads the FAISS index and answers questions using an LCEL chain.

    The chain is built lazily on the first call to ``query()`` to avoid hitting
    the filesystem at import time and to keep unit-test construction cheap.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._chain = None  # built lazily on first query

    def _load_vectorstore(self) -> FAISS:
        """Load the persisted FAISS index from disk via the shared helper."""
        return _load_faiss(self._settings)

    def _build_chain(self) -> None:
        """Construct the LCEL RAG chain and cache it on self._chain.

        Chain structure:
            RunnableParallel(
                context = retriever | _format_docs,
                question = RunnablePassthrough(),
            )
            | prompt
            | ChatOpenAI
            | StrOutputParser
        """
        vectorstore = self._load_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._settings.retriever_k},
        )
        prompt = build_qa_prompt()
        llm = ChatOpenAI(
            model=self._settings.openai_model,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
            temperature=0,
        )

        self._chain = (
            RunnableParallel(
                context=(retriever | _format_docs),
                question=RunnablePassthrough(),
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.debug("RAG chain built successfully.")

    def query(self, question: str) -> str:
        """Answer a question using the RAG chain.

        Builds the chain lazily on first call.

        Args:
            question: Natural language question string.

        Returns:
            str: The model's answer grounded in retrieved document context.

        Raises:
            VectorStoreNotFoundError: If the FAISS index has not been built yet.
            QueryError: If the chain invocation fails for any other reason.
        """
        if self._chain is None:
            self._build_chain()

        logger.info("Querying RAG chain: %r", question)
        try:
            answer = self._chain.invoke(question)
            logger.info("Query answered successfully.")
            return answer
        except VectorStoreNotFoundError:
            raise
        except Exception as exc:
            raise QueryError(f"Chain invocation failed: {exc}") from exc

    def query_with_sources(self, question: str) -> dict:
        """Answer a question and return both the answer and source file paths.

        Performs a direct similarity search to retrieve source metadata, then
        generates the answer via the standard RAG chain.

        Args:
            question: Natural language question string.

        Returns:
            dict: ``{"answer": str, "sources": List[str]}`` where sources are
            unique ``source`` metadata values from the retrieved documents.

        Raises:
            VectorStoreNotFoundError: If the FAISS index has not been built yet.
            QueryError: If the chain invocation fails.
        """
        if self._chain is None:
            self._build_chain()

        answer = self.query(question)

        # Re-load vectorstore to perform a direct similarity search for source metadata
        vectorstore = self._load_vectorstore()
        retrieved_docs = vectorstore.similarity_search(question, k=self._settings.retriever_k)
        sources = list(
            {doc.metadata.get("source", "unknown") for doc in retrieved_docs}
        )

        return {"answer": answer, "sources": sources}


class ConversationalRAGEngine:
    """Multi-turn conversational RAG engine with persistent chat history.

    Uses a history-aware retriever that reformulates follow-up questions into
    standalone queries before retrieval, so references like "tell me more about
    that" resolve correctly even without re-stating the topic.

    Usage::

        engine = ConversationalRAGEngine()
        print(engine.chat("What is Agentic AI?"))
        print(engine.chat("What are its key characteristics?"))  # follow-up works
        engine.clear_history()  # start a fresh session
    """

    def __init__(self, settings: Settings | None = None, retriever=None, cot: bool = False) -> None:
        self._settings = settings or get_settings()
        self._external_retriever = retriever  # optional pre-built retriever
        self._cot = cot
        self._chat_history: List[BaseMessage] = []
        self._conv_chain = None  # built lazily on first chat() call

    def _build_conv_chain(self) -> None:
        """Construct the conversational LCEL chain and cache it.

        Manual LCEL implementation (no langchain.chains helpers required):

          1. If chat_history is non-empty, rewrite the question into a
             standalone question using the contextualize prompt + LLM.
          2. Retrieve documents using the (possibly rewritten) question.
          3. Generate an answer conditioned on context + full chat history.

        When ``cot=True`` the final step uses ``llm.with_structured_output``
        so the chain returns a ``CoTAnswer`` instead of a plain string.
        When an external retriever is supplied it is used directly; otherwise
        a similarity-search retriever is built from the default FAISS index.
        """
        if self._external_retriever is not None:
            retriever = self._external_retriever
            logger.debug("Using externally provided retriever.")
        else:
            vectorstore = _load_faiss(self._settings)
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self._settings.retriever_k},
            )
        llm = ChatOpenAI(
            model=self._settings.openai_model,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
            temperature=0,
        )

        contextualize_chain = build_contextualize_prompt() | llm | StrOutputParser()

        def _get_standalone_question(inputs: dict) -> str:
            """Rewrite follow-up into standalone question when history exists."""
            if inputs.get("chat_history"):
                return contextualize_chain.invoke(
                    {"input": inputs["input"], "chat_history": inputs["chat_history"]}
                )
            return inputs["input"]

        def _retrieve(inputs: dict) -> str:
            standalone_q = inputs["standalone_question"]
            docs = retriever.invoke(standalone_q)
            return _format_docs(docs)

        base = (
            RunnablePassthrough.assign(
                standalone_question=RunnableLambda(_get_standalone_question)
            )
            | RunnablePassthrough.assign(
                context=RunnableLambda(_retrieve)
            )
        )

        if self._cot:
            self._conv_chain = (
                base
                | build_cot_conversational_qa_prompt()
                | llm.with_structured_output(CoTAnswer)
            )
            logger.debug("Conversational RAG chain built (CoT mode).")
        else:
            self._conv_chain = (
                base
                | build_conversational_qa_prompt()
                | llm
                | StrOutputParser()
            )
            logger.debug("Conversational RAG chain built successfully.")

    def chat(self, question: str) -> str:
        """Send a message and get a reply, maintaining full conversation context.

        When ``cot=True`` this delegates to ``chat_with_reasoning`` and returns
        only the final answer string.

        Args:
            question: The user's message (can be a follow-up referencing prior turns).

        Returns:
            str: The assistant's answer grounded in retrieved document context.

        Raises:
            VectorStoreNotFoundError: If the FAISS index has not been built yet.
            QueryError: If the chain invocation fails.
        """
        if self._cot:
            return self.chat_with_reasoning(question).answer

        if self._conv_chain is None:
            self._build_conv_chain()

        logger.info("Conversational query (turn %d): %r", len(self._chat_history) // 2 + 1, question)
        try:
            answer: str = self._conv_chain.invoke(
                {"input": question, "chat_history": self._chat_history}
            )
        except VectorStoreNotFoundError:
            raise
        except Exception as exc:
            raise QueryError(f"Conversational chain invocation failed: {exc}") from exc

        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))
        logger.info("Conversational query answered (history length: %d turns).", len(self._chat_history) // 2)
        return answer

    def chat_with_reasoning(self, question: str) -> CoTAnswer:
        """Send a message and get a structured reply with step-by-step reasoning.

        Requires the engine to be constructed with ``cot=True``.

        Args:
            question: The user's message.

        Returns:
            CoTAnswer: Pydantic model with ``reasoning`` and ``answer`` fields.

        Raises:
            ValueError: If called on an engine that was not built with ``cot=True``.
            VectorStoreNotFoundError: If the FAISS index has not been built yet.
            QueryError: If the chain invocation fails.
        """
        if not self._cot:
            raise ValueError("chat_with_reasoning requires cot=True")

        if self._conv_chain is None:
            self._build_conv_chain()

        logger.info("CoT query (turn %d): %r", len(self._chat_history) // 2 + 1, question)
        try:
            cot_answer: CoTAnswer = self._conv_chain.invoke(
                {"input": question, "chat_history": self._chat_history}
            )
        except VectorStoreNotFoundError:
            raise
        except Exception as exc:
            raise QueryError(f"CoT chain invocation failed: {exc}") from exc

        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=cot_answer.answer))
        logger.info("CoT query answered (history length: %d turns).", len(self._chat_history) // 2)
        return cot_answer

    def clear_history(self) -> None:
        """Reset the chat history to start a fresh session."""
        self._chat_history.clear()
        logger.debug("Chat history cleared.")

    @property
    def history(self) -> List[BaseMessage]:
        """Read-only view of the current chat history."""
        return list(self._chat_history)


def build_mmr_retriever(vectorstore_dir: Path, settings: Settings):
    """Load a FAISS index and return a Maximal Marginal Relevance retriever.

    MMR balances relevance and diversity: ``fetch_k`` candidates are retrieved,
    then ``k`` are selected to maximise coverage of different information.
    ``lambda_mult`` controls the relevance/diversity trade-off (0 = max
    diversity, 1 = pure relevance).

    Args:
        vectorstore_dir: Path to the directory containing the FAISS index.
        settings: Application settings (used for embedding model + MMR params).

    Returns:
        A LangChain retriever configured for MMR search.

    Raises:
        VectorStoreNotFoundError: If the index files are missing.
    """
    vectorstore = _load_faiss(settings, vectorstore_dir=vectorstore_dir)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": settings.mmr_fetch_k,
            "lambda_mult": settings.mmr_lambda_mult,
        },
    )
    logger.debug(
        "MMR retriever built (k=%d, fetch_k=%d, lambda_mult=%.2f)",
        settings.retriever_k,
        settings.mmr_fetch_k,
        settings.mmr_lambda_mult,
    )
    return retriever


def _load_faiss(settings: Settings, vectorstore_dir: Path | None = None) -> FAISS:
    """Shared helper: load the FAISS index from disk.

    Args:
        settings: Application settings (embedding model + credentials).
        vectorstore_dir: Override the index directory. Defaults to
            ``settings.vectorstore_dir``.

    Raises:
        VectorStoreNotFoundError: If the vectorstore directory or index file is missing.
    """
    vectorstore_path = vectorstore_dir or settings.vectorstore_dir
    index_file = vectorstore_path / "index.faiss"

    if not vectorstore_path.exists() or not index_file.exists():
        raise VectorStoreNotFoundError(
            f"Vector store not found at '{vectorstore_path}'. "
            "Run ingestion first: python main.py ingest"
        )

    logger.debug("Loading FAISS index from: %s", vectorstore_path)
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    return FAISS.load_local(
        folder_path=str(vectorstore_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
