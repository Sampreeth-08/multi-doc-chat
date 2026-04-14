import argparse
import sys

from multi_doc_chat.exception.exceptions import MultiDocChatError
from multi_doc_chat.logger.logger import get_logger
from multi_doc_chat.src.ingestion import IngestionPipeline
from multi_doc_chat.src.retrieval import ConversationalRAGEngine, RAGQueryEngine

logger = get_logger(__name__)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Run the ingestion pipeline and print a summary."""
    pipeline = IngestionPipeline()
    result = pipeline.run()
    print(f"Ingestion complete.")
    print(f"  Documents loaded : {result.files_loaded}")
    print(f"  Chunks created   : {result.chunks_created}")
    print(f"  Vector store     : {result.vectorstore_path}")
    if result.errors:
        print(f"  Warnings         : {len(result.errors)} file(s) failed to load")
        for err in result.errors:
            print(f"    - {err}")


def cmd_query(args: argparse.Namespace) -> None:
    """Run a single question through the RAG engine and print the answer."""
    engine = RAGQueryEngine()
    answer = engine.query(args.question)
    print(f"\nAnswer:\n{answer}\n")


def cmd_query_sources(args: argparse.Namespace) -> None:
    """Run a question through the RAG engine and print the answer with sources."""
    engine = RAGQueryEngine()
    result = engine.query_with_sources(args.question)
    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources:")
    for src in result["sources"]:
        print(f"  - {src}")
    print()


def cmd_chat(args: argparse.Namespace) -> None:
    """Run an interactive multi-turn chat session against the indexed documents."""
    engine = ConversationalRAGEngine()
    print("Chat session started. Type 'exit' or press Ctrl-C to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break
        try:
            answer = engine.chat(question)
            print(f"\nAssistant: {answer}\n")
        except MultiDocChatError as exc:
            print(f"Error: {exc}\n", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-doc-chat",
        description="Multi-document RAG chat application powered by OpenAI + FAISS.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest subcommand
    subparsers.add_parser(
        "ingest",
        help="Load, chunk, embed, and store documents from the data/ folder.",
    )

    # query subcommand
    query_parser = subparsers.add_parser(
        "query",
        help="Ask a question and get an answer from the indexed documents.",
    )
    query_parser.add_argument("question", type=str, help="The question to answer.")

    # query-sources subcommand
    query_src_parser = subparsers.add_parser(
        "query-sources",
        help="Ask a question and get an answer with source document references.",
    )
    query_src_parser.add_argument("question", type=str, help="The question to answer.")

    # chat subcommand — interactive multi-turn session
    subparsers.add_parser(
        "chat",
        help="Start an interactive multi-turn chat session against your documents.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            cmd_ingest(args)
        elif args.command == "query":
            cmd_query(args)
        elif args.command == "query-sources":
            cmd_query_sources(args)
        elif args.command == "chat":
            cmd_chat(args)
    except MultiDocChatError as exc:
        logger.error("Application error: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
