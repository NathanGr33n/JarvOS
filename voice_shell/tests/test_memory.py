from pathlib import Path

from voice_shell.src.memory import MemoryStore


class TestMemoryStore:
    def test_add_and_get_recent_turns(self, tmp_path: Path):
        store = MemoryStore(tmp_path / "memory.db", enabled=True)
        store.add_turn("hello", "hi there")
        store.add_turn("what time is it", "it is noon")
        turns = store.get_recent_turns(limit=2)
        assert len(turns) == 2
        assert turns[0].user_text == "hello"
        assert turns[1].assistant_text == "it is noon"
        store.close()

    def test_history_limit_returns_oldest_first_subset(self, tmp_path: Path):
        store = MemoryStore(tmp_path / "memory.db")
        for i in range(5):
            store.add_turn(f"u{i}", f"a{i}")
        turns = store.get_recent_turns(limit=3)
        assert [t.user_text for t in turns] == ["u2", "u3", "u4"]
        store.close()

    def test_facts_set_get_list(self, tmp_path: Path):
        store = MemoryStore(tmp_path / "memory.db")
        store.set_fact("user_name", "Nathan")
        store.set_fact("timezone", "UTC")
        assert store.get_fact("user_name") == "Nathan"
        facts = dict(store.list_facts())
        assert facts["timezone"] == "UTC"
        store.set_fact("user_name", "Nate")
        assert store.get_fact("user_name") == "Nate"
        store.close()

    def test_disabled_store_is_noop(self, tmp_path: Path):
        db = tmp_path / "disabled.db"
        store = MemoryStore(db, enabled=False)
        store.add_turn("a", "b")
        store.set_fact("k", "v")
        assert store.get_recent_turns() == []
        assert store.get_fact("k") is None
        assert not db.exists()
        store.close()

    def test_clear_turns_keeps_facts(self, tmp_path: Path):
        store = MemoryStore(tmp_path / "memory.db")
        store.add_turn("u", "a")
        store.set_fact("pref", "dark")
        store.clear_turns()
        assert store.get_recent_turns() == []
        assert store.get_fact("pref") == "dark"
        store.close()
