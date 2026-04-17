from langchain_core.prompts import ChatPromptTemplate

from multi_doc_chat.prompts.templates import (
    CONTEXTUALIZE_SYSTEM_TEMPLATE,
    COT_SYSTEM_TEMPLATE,
    SYSTEM_TEMPLATE,
    build_contextualize_prompt,
    build_conversational_qa_prompt,
    build_cot_conversational_qa_prompt,
    build_qa_prompt,
)


class TestBuildQaPrompt:
    def test_returns_chat_prompt_template(self):
        prompt = build_qa_prompt()
        assert isinstance(prompt, ChatPromptTemplate)

    def test_has_context_and_question_placeholders(self):
        prompt = build_qa_prompt()
        variables = prompt.input_variables
        assert "context" in variables
        assert "question" in variables

    def test_system_message_contains_key_instruction(self):
        prompt = build_qa_prompt()
        system_content = prompt.messages[0].prompt.template
        assert "context" in system_content.lower()


class TestBuildConversationalQaPrompt:
    def test_returns_chat_prompt_template(self):
        prompt = build_conversational_qa_prompt()
        assert isinstance(prompt, ChatPromptTemplate)

    def test_has_chat_history_placeholder(self):
        prompt = build_conversational_qa_prompt()
        msg_names = [getattr(m, "variable_name", None) for m in prompt.messages]
        assert "chat_history" in msg_names

    def test_has_input_placeholder(self):
        prompt = build_conversational_qa_prompt()
        assert "input" in prompt.input_variables


class TestBuildCotConversationalQaPrompt:
    def test_returns_chat_prompt_template(self):
        prompt = build_cot_conversational_qa_prompt()
        assert isinstance(prompt, ChatPromptTemplate)

    def test_has_chat_history_placeholder(self):
        prompt = build_cot_conversational_qa_prompt()
        msg_names = [getattr(m, "variable_name", None) for m in prompt.messages]
        assert "chat_history" in msg_names

    def test_system_template_instructs_step_by_step_reasoning(self):
        assert "step by step" in COT_SYSTEM_TEMPLATE.lower()

    def test_has_input_placeholder(self):
        prompt = build_cot_conversational_qa_prompt()
        assert "input" in prompt.input_variables


class TestBuildContextualizePrompt:
    def test_returns_chat_prompt_template(self):
        prompt = build_contextualize_prompt()
        assert isinstance(prompt, ChatPromptTemplate)

    def test_has_chat_history_placeholder(self):
        prompt = build_contextualize_prompt()
        msg_names = [getattr(m, "variable_name", None) for m in prompt.messages]
        assert "chat_history" in msg_names

    def test_system_template_says_not_to_answer(self):
        assert "do not answer" in CONTEXTUALIZE_SYSTEM_TEMPLATE.lower()

    def test_has_input_placeholder(self):
        prompt = build_contextualize_prompt()
        assert "input" in prompt.input_variables
