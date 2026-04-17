import pytest
from pydantic import ValidationError

from multi_doc_chat.api.models import ChatRequest, ChatResponse, RunResponse
from multi_doc_chat.model.rag_model import CoTAnswer, IngestionResult, QueryRequest, QueryResponse


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class TestChatRequest:
    def test_valid_question(self):
        req = ChatRequest(question="What is RAG?")
        assert req.question == "What is RAG?"

    def test_missing_question_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest()

    def test_empty_string_is_accepted(self):
        req = ChatRequest(question="")
        assert req.question == ""


class TestChatResponse:
    def test_all_fields_accessible(self):
        resp = ChatResponse(
            session_id="s1",
            question="Q?",
            reasoning="Step 1...",
            answer="The answer.",
        )
        assert resp.session_id == "s1"
        assert resp.question == "Q?"
        assert resp.reasoning == "Step 1..."
        assert resp.answer == "The answer."

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ChatResponse(session_id="s1", question="Q?", reasoning="r")


class TestRunResponse:
    def test_files_count_defaults_to_zero(self):
        resp = RunResponse(
            session_id="s1",
            started_at="2024-01-01T00:00:00",
            files_loaded=3,
            chunks_created=12,
        )
        assert resp.files_count == 0

    def test_explicit_files_count(self):
        resp = RunResponse(
            session_id="s1",
            started_at="2024-01-01T00:00:00",
            files_count=5,
            files_loaded=5,
            chunks_created=20,
        )
        assert resp.files_count == 5


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class TestCoTAnswer:
    def test_fields_accessible(self):
        cot = CoTAnswer(reasoning="I reasoned.", answer="Final answer.")
        assert cot.reasoning == "I reasoned."
        assert cot.answer == "Final answer."

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            CoTAnswer(reasoning="only reasoning, no answer")


class TestIngestionResult:
    def test_errors_defaults_to_empty_list(self):
        result = IngestionResult(
            files_loaded=2, chunks_created=10, vectorstore_path="/vs"
        )
        assert result.errors == []

    def test_explicit_errors(self):
        result = IngestionResult(
            files_loaded=1,
            chunks_created=0,
            vectorstore_path="/vs",
            errors=["bad.pdf: corrupt"],
        )
        assert len(result.errors) == 1


class TestQueryRequest:
    def test_question_stored(self):
        qr = QueryRequest(question="Test?")
        assert qr.question == "Test?"


class TestQueryResponse:
    def test_sources_defaults_to_empty_list(self):
        qr = QueryResponse(question="Q?", answer="A.")
        assert qr.sources == []

    def test_explicit_sources(self):
        qr = QueryResponse(question="Q?", answer="A.", sources=["doc1.pdf"])
        assert qr.sources == ["doc1.pdf"]
