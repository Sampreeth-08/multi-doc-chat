from unittest.mock import MagicMock

import pytest

from multi_doc_chat.api import session_store


@pytest.fixture(autouse=True)
def clear_store():
    """Isolate each test by clearing the in-memory store before and after."""
    session_store._store.clear()
    yield
    session_store._store.clear()


def _make_entry():
    session = MagicMock()
    engine = MagicMock()
    return session, engine


class TestSessionStoreSave:
    def test_save_makes_entry_retrievable(self):
        session, engine = _make_entry()
        session_store.save("sid-1", session, engine)
        result = session_store.get("sid-1")
        assert result is not None
        assert result == (session, engine)

    def test_save_overwrites_existing_entry(self):
        session1, engine1 = _make_entry()
        session2, engine2 = _make_entry()
        session_store.save("sid-1", session1, engine1)
        session_store.save("sid-1", session2, engine2)
        s, e = session_store.get("sid-1")
        assert s is session2
        assert e is engine2


class TestSessionStoreGet:
    def test_get_returns_none_for_unknown_id(self):
        assert session_store.get("does-not-exist") is None

    def test_get_returns_correct_entry(self):
        session, engine = _make_entry()
        session_store.save("abc", session, engine)
        s, e = session_store.get("abc")
        assert s is session
        assert e is engine


class TestSessionStoreRemove:
    def test_remove_deletes_entry(self):
        session, engine = _make_entry()
        session_store.save("to-remove", session, engine)
        session_store.remove("to-remove")
        assert session_store.get("to-remove") is None

    def test_remove_missing_id_does_not_raise(self):
        session_store.remove("ghost-id")  # must not raise

    def test_remove_does_not_affect_other_entries(self):
        s1, e1 = _make_entry()
        s2, e2 = _make_entry()
        session_store.save("keep", s1, e1)
        session_store.save("delete", s2, e2)
        session_store.remove("delete")
        assert session_store.get("keep") is not None
        assert session_store.get("delete") is None
