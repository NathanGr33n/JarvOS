from pathlib import Path

from voice_shell.src.actions.audit import ActionAuditLog


class TestActionAuditLog:
    def test_disabled_store_is_noop(self, tmp_path: Path):
        log = ActionAuditLog(tmp_path / "audit.db", enabled=False)
        log.record("ls", status="success", detail="ok")
        assert log.count() == 0
        assert log.list_entries() == []
        log.close()

    def test_record_and_list_newest_first(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        log = ActionAuditLog(db, enabled=True)
        log.record("ls", argument=".", status="success", detail="files", user_transcript="list files")
        log.record(
            "write_file",
            argument="a|b",
            status="cancelled",
            detail="user declined confirmation",
            confirmed=False,
            user_transcript="write note",
        )
        assert log.count() == 2
        entries = log.list_entries(limit=10)
        assert [e.action_name for e in entries] == ["write_file", "ls"]
        assert entries[0].status == "cancelled"
        assert entries[1].status == "success"
        assert entries[1].user_transcript == "list files"
        log.close()

    def test_filter_by_name_and_status(self, tmp_path: Path):
        log = ActionAuditLog(tmp_path / "audit.db")
        log.record("time", status="success")
        log.record("app:firefox", status="error", detail="Cannot launch")
        log.record("app:firefox", status="success", confirmed=True)

        only_ff = log.list_entries(action_name="app:firefox")
        assert len(only_ff) == 2
        assert all(e.action_name == "app:firefox" for e in only_ff)

        errors = log.list_entries(status="error")
        assert len(errors) == 1
        assert errors[0].detail == "Cannot launch"
        log.close()

    def test_persists_across_reopen(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        log = ActionAuditLog(db)
        log.record("pwd", status="success", detail="/tmp")
        log.close()

        log2 = ActionAuditLog(db)
        assert log2.count() == 1
        entry = log2.list_entries(limit=1)[0]
        assert entry.action_name == "pwd"
        assert entry.detail == "/tmp"
        log2.close()