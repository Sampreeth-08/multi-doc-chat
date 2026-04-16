from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_TEMPLATE = """\
You are a helpful assistant that answers questions strictly based on the \
provided context documents. If the answer cannot be found in the context, \
say "I don't have enough information to answer that question." \
Do not fabricate information or draw on knowledge outside the provided context.

Context:
{context}
"""

COT_SYSTEM_TEMPLATE = """\
You are a helpful assistant that answers questions strictly based on the \
provided context documents. Think through the problem step by step before \
giving your final answer. If the answer cannot be found in the context, \
say so in your reasoning and answer "I don't have enough information to \
answer that question." Do not fabricate information.

Context:
{context}
"""

# Used by the history-aware retriever to rewrite follow-up questions into
# standalone questions that make sense without the chat history.
CONTEXTUALIZE_SYSTEM_TEMPLATE = """\
Given a chat history and the latest user question which might reference \
context in the chat history, formulate a standalone question which can be \
understood without the chat history. Do NOT answer the question — just \
reformulate it if needed, otherwise return it as-is.
"""


def build_qa_prompt() -> ChatPromptTemplate:
    """QA prompt for the single-turn RAG chain.

    Placeholders: ``{context}``, ``{question}``.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            ("human", "Question: {question}"),
        ]
    )


def build_conversational_qa_prompt() -> ChatPromptTemplate:
    """QA prompt for the conversational RAG chain.

    Placeholders: ``{context}``, ``{chat_history}`` (MessagesPlaceholder),
    ``{input}`` (the current human turn).
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def build_cot_conversational_qa_prompt() -> ChatPromptTemplate:
    """CoT QA prompt for the conversational RAG chain.

    Instructs the LLM to reason step by step. Used with
    ``llm.with_structured_output(CoTAnswer)`` to produce a
    ``{reasoning, answer}`` response.

    Placeholders: ``{context}``, ``{chat_history}``, ``{input}``.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", COT_SYSTEM_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def build_contextualize_prompt() -> ChatPromptTemplate:
    """Prompt that instructs the LLM to rewrite a follow-up question into a
    standalone question given the chat history.

    Used by ``create_history_aware_retriever``.
    Placeholders: ``{chat_history}`` (MessagesPlaceholder), ``{input}``.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
