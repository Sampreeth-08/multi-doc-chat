Plan: Multi-Document RAG Application
Context
Build a production-quality RAG application inside the existing multi_doc_chat/ module scaffold. The project already has a LangChain+FAISS prototype in notebook/RAG.ipynb; this plan formalizes that into a proper package with logging, custom exceptions, data ingestion, a RAG query engine, and a pytest test suite. The OpenAI API key and base URL live in .env. Only the CLI/library interface is needed (no web server).

1. Dependency Update — pyproject.toml
Add runtime deps under [project] dependencies and dev deps under [dependency-groups] dev:

dependencies = [
    "python-dotenv>=1.2.2",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-community>=0.3.0",
    "langchain-text-splitters>=0.3.0",
    "faiss-cpu>=1.9.0",
    "tiktoken>=0.7.0",
    "pypdf>=5.0.0",
    "python-docx>=1.1.0",
    "docx2txt>=0.8",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.14.0",
]
Run uv sync after editing to install and lock new packages.

2. File Structure
multi_doc_chat/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py          ← frozen Settings dataclass + get_settings()
├── exception/
│   ├── __init__.py
│   └── exceptions.py        ← exception hierarchy
├── logger/
│   ├── __init__.py
│   └── logger.py            ← get_logger() factory
├── model/
│   ├── __init__.py
│   └── rag_model.py         ← QueryRequest, QueryResponse, IngestionResult dataclasses
├── prompts/
│   ├── __init__.py
│   └── templates.py         ← build_qa_prompt()
├── src/
│   ├── __init__.py
│   ├── ingestion.py         ← DocumentLoader, DocumentChunker, VectorStoreBuilder, IngestionPipeline
│   └── retrieval.py         ← RAGQueryEngine
└── utils/
    ├── __init__.py
    └── file_utils.py        ← iter_supported_files(), get_extension(), ensure_dir()
test/
├── __init__.py
├── conftest.py              ← shared fixtures (sample_settings, sample_documents)
├── test_file_utils.py
├── test_exceptions.py
├── test_ingestion.py
└── test_retrieval.py
logs/                        ← created at runtime
vectorstore/                 ← created at runtime by FAISS.save_local
main.py                      ← CLI entry point (ingest / query subcommands)
3. Module Specifications
multi_doc_chat/logger/logger.py
get_logger(name, level=DEBUG) -> Logger
Creates logs/ dir on first call
RotatingFileHandler → logs/multi_doc_chat.log (5 MB, 3 backups, DEBUG)
StreamHandler → console (WARNING only, keeps CLI clean)
Idempotent: handlers added once per named logger
multi_doc_chat/exception/exceptions.py
MultiDocChatError (base)
├── ConfigurationError        ← missing/invalid env vars
├── DocumentLoadError(file_path, reason)
│   └── UnsupportedFileTypeError
├── VectorStoreError
│   └── VectorStoreNotFoundError
└── QueryError
multi_doc_chat/config/settings.py
@dataclass(frozen=True) class Settings with fields: openai_api_key, openai_base_url, openai_model (default gpt-4o-mini), openai_embedding_model (default text-embedding-3-small), chunk_size (default 1000), chunk_overlap (default 200), retriever_k (default 4), data_dir: Path, vectorstore_dir: Path
get_settings() -> Settings — singleton, raises ConfigurationError if OPENAI_API_KEY missing
All overridable via env vars (OPENAI_MODEL, CHUNK_SIZE, etc.)
multi_doc_chat/prompts/templates.py
SYSTEM_TEMPLATE — instructs model to answer only from context; says "I don't have enough information" if not found
HUMAN_TEMPLATE = "Question: {question}"
build_qa_prompt() -> ChatPromptTemplate using from_messages([system, human])
multi_doc_chat/utils/file_utils.py
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}
iter_supported_files(data_dir: Path) -> Generator[Path] — recursive, raises FileNotFoundError if dir missing
get_extension(file: Path) -> str — lowercase extension
ensure_dir(path: Path) -> None — mkdir parents
multi_doc_chat/src/ingestion.py
Four classes:

DocumentLoader

load_file(file_path: Path) -> List[Document]
.pdf → PyPDFLoader
.txt → TextLoader
.docx / .doc → Docx2txtLoader
else → UnsupportedFileTypeError
wraps loader errors in DocumentLoadError
load_directory(data_dir=None) -> List[Document]
Calls iter_supported_files, calls load_file per file
Logs warning on per-file error, continues; raises DocumentLoadError if zero docs loaded
DocumentChunker

Uses RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
chunk(documents) -> List[Document] — raises ValueError if input empty
VectorStoreBuilder

Creates OpenAIEmbeddings(model, api_key, base_url)
build_and_save(chunks) -> None
FAISS.from_documents(chunks, embeddings) wrapped in VectorStoreError
ensure_dir(vectorstore_dir) then vectorstore.save_local(str(vectorstore_dir))
IngestionPipeline

run(data_dir=None) -> IngestionResult
orchestrates: loader → chunker → builder
returns IngestionResult(files_loaded, chunks_created, vectorstore_path, errors)
multi_doc_chat/src/retrieval.py
RAGQueryEngine

Lazy _chain = None; built on first query() call via _build_chain()
_load_vectorstore() -> FAISS
FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
Raises VectorStoreNotFoundError if dir missing
_build_chain() — LCEL:
RunnableParallel(context=(retriever | _format_docs), question=RunnablePassthrough())
| build_qa_prompt()
| ChatOpenAI(model, api_key, base_url, temperature=0)
| StrOutputParser()
query(question: str) -> str — invokes chain, wraps errors in QueryError
query_with_sources(question: str) -> dict — returns {"answer": str, "sources": [...]}
multi_doc_chat/model/rag_model.py
@dataclass class QueryRequest: question: str
@dataclass class QueryResponse: question: str; answer: str; sources: List[str]
@dataclass class IngestionResult: files_loaded: int; chunks_created: int; vectorstore_path: str; errors: List[str]
main.py (Updated)
CLI with two subcommands:

python main.py ingest → runs IngestionPipeline().run(); prints summary
python main.py query "..." → runs RAGQueryEngine().query(question); prints answer
Top-level catches MultiDocChatError, logs error, exits with code 1
4. Testing Strategy
test/conftest.py
Fixtures:

sample_settings — Settings with dummy key (sk-test-key), tmp_path-based dirs
sample_documents — two Document objects
mock_faiss_vectorstore — MagicMock with .as_retriever() returning a mock retriever
test/test_file_utils.py (no mocking, uses tmp_path)
Yields only .pdf, .txt, .doc, .docx; skips .csv
Recursive traversal finds nested files
FileNotFoundError on nonexistent dir
test/test_exceptions.py
Hierarchy assertions (issubclass)
str() of DocumentLoadError contains path and reason
test/test_ingestion.py
DocumentLoader.load_file with real .txt file in tmp_path
UnsupportedFileTypeError on .csv
load_directory continues past one failing file (mock loader to raise)
DocumentChunker.chunk with real splitter — output chunk count and sizes
ValueError on empty documents list
VectorStoreBuilder.build_and_save — mock OpenAIEmbeddings, FAISS.from_documents, FAISS.save_local
test/test_retrieval.py
RAGQueryEngine.query — mock FAISS.load_local, ChatOpenAI, assert string returned
VectorStoreNotFoundError when vectorstore dir missing
Chain built only once across two query() calls
Run all tests: uv run pytest test/ -v

5. Critical Files to Create/Modify
File	Action
pyproject.toml	Update dependencies
multi_doc_chat/__init__.py	Create (empty)
multi_doc_chat/config/__init__.py	Create (empty)
multi_doc_chat/config/settings.py	Create
multi_doc_chat/exception/__init__.py	Create (empty)
multi_doc_chat/exception/exceptions.py	Create
multi_doc_chat/logger/__init__.py	Create (empty)
multi_doc_chat/logger/logger.py	Create
multi_doc_chat/model/__init__.py	Create (empty)
multi_doc_chat/model/rag_model.py	Create
multi_doc_chat/prompts/__init__.py	Create (empty)
multi_doc_chat/prompts/templates.py	Create
multi_doc_chat/src/__init__.py	Create (empty)
multi_doc_chat/src/ingestion.py	Create
multi_doc_chat/src/retrieval.py	Create
multi_doc_chat/utils/__init__.py	Create (empty)
multi_doc_chat/utils/file_utils.py	Create
test/__init__.py	Create (empty)
test/conftest.py	Create
test/test_file_utils.py	Create
test/test_exceptions.py	Create
test/test_ingestion.py	Create
test/test_retrieval.py	Create
main.py	Rewrite
6. Verification
uv sync — installs all deps without error
uv run python main.py ingest — processes data/Agentic AI.txt; vectorstore/ dir appears with index.faiss + index.pkl
uv run python main.py query "What are the key characteristics of Agentic AI?" — prints a coherent answer
uv run pytest test/ -v — all tests pass (ingestion/retrieval tests use mocks, no real API calls)
cat logs/multi_doc_chat.log — shows DEBUG-level entries for load, chunk, embed, query steps
Add Comment