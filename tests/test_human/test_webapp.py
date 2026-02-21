"""Tests for human evaluation webapp."""

import pytest

# Skip all tests if Flask is not installed
pytest.importorskip("flask")

from agentick.human.webapp import ShowcaseWebApp, _action_name_to_int, create_app


def test_webapp_initialization():
    """Test that webapp can be initialized."""
    webapp = ShowcaseWebApp(port=5001)
    assert webapp is not None
    assert webapp.app is not None
    assert webapp.port == 5001


def test_create_app():
    """Test create_app factory function."""
    app = create_app()
    assert app is not None
    assert hasattr(app, "route")


def test_webapp_routes():
    """Test that webapp has required routes."""
    webapp = ShowcaseWebApp()
    app = webapp.app

    routes = [rule.rule for rule in app.url_map.iter_rules()]

    assert "/" in routes
    assert "/api/task_descriptions" in routes
    assert "/api/start_task" in routes
    assert "/api/step" in routes
    assert "/api/reset" in routes
    assert "/api/quit" in routes
    assert "/gallery/<filename>" in routes or any("gallery" in r for r in routes)


def test_webapp_index_page():
    """Test that index page loads."""
    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"Agentick" in response.data


def test_task_descriptions_endpoint():
    """Test task descriptions API includes gallery_gif field."""
    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    response = client.get("/api/task_descriptions")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0]
    assert "gallery_gif" in data[0]


def test_start_task_endpoint():
    """Test starting a task via API with multi-modal response."""
    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    response = client.post(
        "/api/start_task",
        json={"task_id": "GoToGoal-v0", "difficulty": "easy", "seed": 42},
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "session_id" in data
    assert "ascii" in data
    assert "language" in data
    assert "pixel" in data
    assert data["ascii"]  # non-empty
    assert data["language"]  # non-empty
    assert data["pixel"].startswith("data:image/png;base64,")


def test_step_endpoint_without_session():
    """Test step endpoint without active session."""
    from agentick.human import webapp as webapp_mod

    webapp_mod._sessions.clear()  # ensure no stale sessions
    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    response = client.post(
        "/api/step", json={"action": "move_up"}, content_type="application/json"
    )
    assert response.status_code == 400


def test_full_task_workflow():
    """Test complete workflow: start -> step -> reset -> quit with multi-modal obs."""
    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    # Start task
    response = client.post(
        "/api/start_task",
        json={"task_id": "GoToGoal-v0", "difficulty": "easy"},
        content_type="application/json",
    )
    assert response.status_code == 200
    start_data = response.get_json()
    session_id = start_data["session_id"]

    # Take a step — verify multi-modal response
    response = client.post(
        "/api/step",
        json={"action": "move_down", "session_id": session_id},
        content_type="application/json",
    )
    assert response.status_code == 200
    step_data = response.get_json()
    assert "ascii" in step_data
    assert "language" in step_data
    assert "pixel" in step_data
    assert "reward" in step_data
    assert "steps" in step_data

    # Reset task
    response = client.post(
        "/api/reset",
        json={"session_id": session_id},
        content_type="application/json",
    )
    assert response.status_code == 200
    reset_data = response.get_json()
    assert reset_data["success"] is True
    assert "ascii" in reset_data
    assert "language" in reset_data
    assert "pixel" in reset_data

    # Quit task
    response = client.post(
        "/api/quit",
        json={"session_id": session_id},
        content_type="application/json",
    )
    assert response.status_code == 200
    quit_data = response.get_json()
    assert quit_data["success"] is True


def test_gallery_route():
    """Test gallery route serves GIF files."""
    from pathlib import Path

    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    # Create a temporary gallery dir with a test GIF
    gallery_dir = Path(__file__).resolve().parent.parent.parent / "gallery"
    if gallery_dir.is_dir() and (gallery_dir / "GoToGoal-v0.gif").exists():
        response = client.get("/gallery/GoToGoal-v0.gif")
        assert response.status_code == 200
        assert response.content_type in ("image/gif", "application/octet-stream")
    else:
        # Gallery dir may not exist in CI; just check route is registered
        response = client.get("/gallery/nonexistent.gif")
        assert response.status_code == 404


def test_action_mapping():
    """Test that action names map correctly."""
    assert _action_name_to_int("noop") == 0
    assert _action_name_to_int("move_up") == 1
    assert _action_name_to_int("move_down") == 2
    assert _action_name_to_int("move_left") == 3
    assert _action_name_to_int("move_right") == 4
    assert _action_name_to_int("pickup") == 5
    assert _action_name_to_int("drop") == 6
    assert _action_name_to_int("use") == 7
    assert _action_name_to_int("interact") == 8
    assert _action_name_to_int("unknown") == 0


def test_action_step_multiple():
    """Test stepping with various actions."""
    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    response = client.post(
        "/api/start_task",
        json={"task_id": "GoToGoal-v0", "difficulty": "easy"},
        content_type="application/json",
    )
    session_id = response.get_json()["session_id"]

    actions = ["noop", "move_up", "move_down", "move_left", "move_right", "pickup"]

    for action in actions:
        response = client.post(
            "/api/step",
            json={"action": action, "session_id": session_id},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()

        if data.get("terminated") or data.get("truncated"):
            # Restart for next action test
            response = client.post(
                "/api/start_task",
                json={"task_id": "GoToGoal-v0", "difficulty": "easy"},
                content_type="application/json",
            )
            session_id = response.get_json()["session_id"]
