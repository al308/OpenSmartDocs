from pathlib import Path

from onedrive_ollama_pipeline.state_store import StateStore


def test_state_store_roundtrip(tmp_path: Path):
    store_path = tmp_path / "state.sqlite3"
    store = StateStore(store_path)
    assert not store.is_processed("abc")

    store.record_success(item_id="abc", filename="file.pdf", model="demo", metadata={"k": "v"})
    reloaded = StateStore(store_path)
    assert reloaded.is_processed("abc")

    reloaded.record_failure(item_id="def", filename="file2.pdf", model="demo", error="bang")
    assert not reloaded.is_processed("def")
    recent = reloaded.recent_items()
    assert recent[0]["status"] in {"success", "failed"}
