import os
from pathlib import Path
from unittest.mock import patch

import pytest

import multi_doc_chat.config.settings as settings_module
from multi_doc_chat.config.settings import Settings, get_settings
from multi_doc_chat.exception.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before and after each test."""
    settings_module._settings_instance = None
    yield
    settings_module._settings_instance = None


_VALID_ENV = {
    "OPENAI_API_KEY": "sk-test-key",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "OPENAI_MODEL": "gpt-4o-mini",
    "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
    "CHUNK_SIZE": "500",
    "CHUNK_OVERLAP": "50",
    "RETRIEVER_K": "4",
    "MMR_FETCH_K": "15",
    "MMR_LAMBDA_MULT": "0.6",
}


class TestGetSettings:
    def test_raises_configuration_error_when_api_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure OPENAI_API_KEY is not set
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
                get_settings()

    def test_returns_settings_with_valid_env(self):
        with patch.dict(os.environ, _VALID_ENV, clear=False):
            settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.openai_api_key == "sk-test-key"
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 50
        assert settings.retriever_k == 4
        assert settings.mmr_fetch_k == 15
        assert settings.mmr_lambda_mult == pytest.approx(0.6)

    def test_returns_singleton_on_repeated_calls(self):
        with patch.dict(os.environ, _VALID_ENV, clear=False):
            s1 = get_settings()
            s2 = get_settings()
        assert s1 is s2

    def test_default_model_when_not_set(self):
        env = {k: v for k, v in _VALID_ENV.items() if k != "OPENAI_MODEL"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("OPENAI_MODEL", None)
            settings = get_settings()
        assert settings.openai_model == "gpt-4o-mini"

    def test_default_base_url_when_not_set(self):
        env = {k: v for k, v in _VALID_ENV.items() if k != "OPENAI_BASE_URL"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("OPENAI_BASE_URL", None)
            settings = get_settings()
        assert settings.openai_base_url == "https://api.openai.com/v1"

    def test_dirs_are_path_objects(self):
        with patch.dict(os.environ, _VALID_ENV, clear=False):
            settings = get_settings()
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.vectorstore_dir, Path)
        assert isinstance(settings.sessions_dir, Path)

    def test_raises_when_api_key_is_whitespace_only(self):
        with patch.dict(os.environ, {**_VALID_ENV, "OPENAI_API_KEY": "   "}, clear=False):
            with pytest.raises(ConfigurationError):
                get_settings()
