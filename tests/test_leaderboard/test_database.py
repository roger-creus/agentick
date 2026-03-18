"""Tests for leaderboard database (thin JSON I/O wrapper)."""

from agentick.leaderboard.database import add_entry, get_entries, load_entries, save_entries


def test_load_entries_empty(tmp_path):
    """Test loading from non-existent file returns empty list."""
    entries = load_entries(tmp_path / "does_not_exist.json")
    assert entries == []


def test_save_and_load_roundtrip(tmp_path):
    """Test save then load roundtrip."""
    path = tmp_path / "entries.json"
    entries = [{"agent_name": "test", "scores": {"agentick_score": 0.5}}]
    save_entries(entries, path)
    loaded = load_entries(path)
    assert len(loaded) == 1
    assert loaded[0]["agent_name"] == "test"


def test_add_entry(tmp_path):
    """Test adding an entry."""
    path = tmp_path / "entries.json"
    entry = {
        "agent_name": "TestAgent",
        "author": "Tester",
        "description": "A test",
        "agent_type": "other",
        "observation_mode": "ascii",
        "harness": "",
        "model": "test",
        "open_weights": False,
        "date": "2026-01-01",
        "scores": {
            "agentick_score": 0.42,
            "agentick_score_ci": [0.40, 0.44],
            "per_category": {},
            "per_task": {},
        },
    }
    add_entry(entry, path)
    loaded = load_entries(path)
    assert len(loaded) == 1
    assert loaded[0]["agent_name"] == "TestAgent"


def test_get_entries_filter(tmp_path):
    """Test get_entries with suite_name filter."""
    path = tmp_path / "entries.json"
    e1 = {
        "agent_name": "A",
        "author": "x",
        "description": "d",
        "agent_type": "rl",
        "observation_mode": "state_dict",
        "harness": "",
        "model": "m",
        "open_weights": True,
        "date": "2026-01-01",
        "scores": {
            "agentick_score": 0.5,
            "agentick_score_ci": [0.4, 0.6],
            "per_category": {},
            "per_task": {},
        },
        "metadata": {"suite_name": "agentick-full-v2"},
    }
    e2 = {**e1, "agent_name": "B", "metadata": {"suite_name": "other-suite"}}
    save_entries([e1, e2], path)

    full = get_entries(suite_name="agentick-full-v2", path=path)
    assert len(full) == 1
    assert full[0]["agent_name"] == "A"

    other = get_entries(suite_name="other-suite", path=path)
    assert len(other) == 1
    assert other[0]["agent_name"] == "B"

    all_entries = get_entries(path=path)
    assert len(all_entries) == 2
