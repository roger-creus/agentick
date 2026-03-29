"""Tests for human play webapp."""

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
    assert "/api/tasks" in routes
    assert "/api/start" in routes
    assert "/api/step" in routes
    assert "/api/reset" in routes
    assert "/api/quit" in routes


def test_webapp_index_page():
    """Test that index page loads."""
    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"Agentick" in response.data


def test_tasks_endpoint():
    """Test tasks API returns task list with category."""
    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    response = client.get("/api/tasks")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0]
    assert "category" in data[0]


def test_start_task_endpoint():
    """Test starting a task via API with multi-modal response."""
    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    response = client.post(
        "/api/start",
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
    assert data["pixel"].startswith("data:image/")


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
        "/api/start",
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


def test_action_mapping():
    """Test that action names map to correct ActionType integers."""
    assert _action_name_to_int("noop") == 0
    assert _action_name_to_int("move_up") == 1
    assert _action_name_to_int("move_down") == 2
    assert _action_name_to_int("move_left") == 3
    assert _action_name_to_int("move_right") == 4
    assert _action_name_to_int("interact") == 5
    assert _action_name_to_int("rotate_left") == 6
    assert _action_name_to_int("rotate_right") == 7
    assert _action_name_to_int("move_forward") == 8
    assert _action_name_to_int("unknown") == 0


def test_action_step_multiple():
    """Test stepping with various actions."""
    webapp = ShowcaseWebApp()
    client = webapp.app.test_client()

    response = client.post(
        "/api/start",
        json={"task_id": "GoToGoal-v0", "difficulty": "easy"},
        content_type="application/json",
    )
    session_id = response.get_json()["session_id"]

    actions = ["noop", "move_up", "move_down", "move_left", "move_right", "interact"]

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
                "/api/start",
                json={"task_id": "GoToGoal-v0", "difficulty": "easy"},
                content_type="application/json",
            )
            session_id = response.get_json()["session_id"]
