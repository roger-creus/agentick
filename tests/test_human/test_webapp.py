"""Tests for human evaluation webapp."""

import pytest

# Skip all tests if Flask is not installed
pytest.importorskip("flask")

from agentick.human.webapp import HumanEvaluationWebApp, create_app


def test_webapp_initialization():
    """Test that webapp can be initialized."""
    webapp = HumanEvaluationWebApp(output_dir="test_output", port=5001)
    assert webapp is not None
    assert webapp.app is not None
    assert webapp.port == 5001


def test_create_app():
    """Test create_app factory function."""
    app = create_app(output_dir="test_output")
    assert app is not None
    assert hasattr(app, "route")


def test_webapp_routes():
    """Test that webapp has required routes."""
    webapp = HumanEvaluationWebApp(output_dir="test_output")
    app = webapp.app

    # Get all route rules
    routes = [rule.rule for rule in app.url_map.iter_rules()]

    # Check required routes
    assert "/" in routes
    assert "/api/start_task" in routes
    assert "/api/step" in routes
    assert "/api/reset" in routes
    assert "/api/quit" in routes


def test_webapp_index_page():
    """Test that index page loads."""
    webapp = HumanEvaluationWebApp(output_dir="test_output")
    client = webapp.app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"Agentick Human Evaluation" in response.data


def test_start_task_endpoint():
    """Test starting a task via API."""
    webapp = HumanEvaluationWebApp(output_dir="test_output")
    client = webapp.app.test_client()

    # Test without task_id (should fail)
    response = client.post("/api/start_task", json={}, content_type="application/json")
    assert response.status_code == 400

    # Test with valid task_id
    response = client.post(
        "/api/start_task",
        json={"task_id": "GoToGoal-v0", "difficulty": "easy"},
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "session_id" in data
    assert data["task_id"] == "GoToGoal-v0"
    assert "render" in data


def test_step_endpoint_without_session():
    """Test step endpoint without active session."""
    webapp = HumanEvaluationWebApp(output_dir="test_output")
    client = webapp.app.test_client()

    response = client.post("/api/step", json={"action": "move_up"}, content_type="application/json")
    assert response.status_code == 400


def test_full_task_workflow():
    """Test complete workflow: start -> step -> reset -> quit."""
    webapp = HumanEvaluationWebApp(output_dir="test_output")
    client = webapp.app.test_client()

    # Start task
    with client.session_transaction() as session:
        session["session_id"] = "test_session"

    response = client.post(
        "/api/start_task",
        json={"task_id": "GoToGoal-v0", "difficulty": "easy"},
        content_type="application/json",
    )
    assert response.status_code == 200
    start_data = response.get_json()
    session_id = start_data["session_id"]

    # Take a step
    with client.session_transaction() as session:
        session["session_id"] = session_id

    response = client.post(
        "/api/step", json={"action": "move_down"}, content_type="application/json"
    )
    assert response.status_code == 200
    step_data = response.get_json()
    assert "render" in step_data

    # Reset task
    response = client.post("/api/reset", json={}, content_type="application/json")
    assert response.status_code == 200
    reset_data = response.get_json()
    assert reset_data["success"] is True

    # Quit task
    response = client.post("/api/quit", json={}, content_type="application/json")
    assert response.status_code == 200
    quit_data = response.get_json()
    assert quit_data["success"] is True


def test_action_mapping():
    """Test that action names map correctly."""
    webapp = HumanEvaluationWebApp(output_dir="test_output")
    client = webapp.app.test_client()

    # Start task
    response = client.post(
        "/api/start_task",
        json={"task_id": "GoToGoal-v0", "difficulty": "easy"},
        content_type="application/json",
    )
    session_id = response.get_json()["session_id"]

    # Test different actions
    actions_to_test = ["noop", "move_up", "move_down", "move_left", "move_right", "pickup"]

    for action in actions_to_test:
        with client.session_transaction() as session:
            session["session_id"] = session_id

        response = client.post(
            "/api/step", json={"action": action}, content_type="application/json"
        )
        assert response.status_code == 200
        data = response.get_json()

        # If episode ended, restart
        if data.get("terminated") or data.get("truncated"):
            response = client.post(
                "/api/start_task",
                json={"task_id": "GoToGoal-v0", "difficulty": "easy"},
                content_type="application/json",
            )
            session_id = response.get_json()["session_id"]
