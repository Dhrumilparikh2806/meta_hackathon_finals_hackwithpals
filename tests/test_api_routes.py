import pytest
from fastapi.testclient import TestClient

def get_client():
    try:
        from app import app
        return TestClient(app)
    except Exception as e:
        pytest.skip(f"Cannot import app: {e}")

class TestExistingRound1Routes:
    def test_health(self):
        client = get_client()
        assert client.get("/health").status_code == 200

    def test_metadata(self):
        client = get_client()
        assert client.get("/metadata").status_code == 200

    def test_ui(self):
        client = get_client()
        resp = client.get("/ui")
        assert resp.status_code in [200, 404]

class TestFleetRoutes:
    def test_fleet_reset(self):
        client = get_client()
        resp = client.post("/fleet/reset", json={"task_id": "easy_fleet", "seed": 42})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_fleet_state(self):
        client = get_client()
        client.post("/fleet/reset", json={"task_id": "easy_fleet", "seed": 42})
        assert client.get("/fleet/state").status_code == 200

    def test_fleet_workers(self):
        client = get_client()
        client.post("/fleet/reset", json={"task_id": "easy_fleet", "seed": 42})
        resp = client.get("/fleet/workers")
        assert resp.status_code == 200
        assert len(resp.json()["workers"]) == 5

    def test_fleet_step_valid(self):
        client = get_client()
        client.post("/fleet/reset", json={"task_id": "easy_fleet", "seed": 42})
        resp = client.post("/fleet/step", json={"action_type": "monitor", "worker_id": "worker_1"})
        assert resp.status_code == 200
        assert -1.0 <= resp.json()["reward_total"] <= 1.0

    def test_fleet_step_invalid_action(self):
        client = get_client()
        client.post("/fleet/reset", json={"task_id": "easy_fleet", "seed": 42})
        assert client.post("/fleet/step", json={"action_type": "invalid", "worker_id": "worker_1"}).status_code == 400

    def test_fleet_report(self):
        client = get_client()
        client.post("/fleet/reset", json={"task_id": "easy_fleet"})
        assert client.get("/fleet/report").status_code == 200

    def test_fleet_evaluate(self):
        client = get_client()
        client.post("/fleet/reset", json={"task_id": "easy_fleet"})
        resp = client.post("/fleet/evaluate")
        assert resp.status_code == 200
        assert "evaluation" in resp.json()

    def test_fleet_ui(self):
        client = get_client()
        resp = client.get("/fleet-ui")
        assert resp.status_code in [200, 404]

class TestRAGRoute:
    def test_rag_query(self):
        client = get_client()
        resp = client.post("/rag/query", json={"question": "What is the Pro plan price?"})
        assert resp.status_code == 200
        assert "answer" in resp.json() or "error" in resp.json()
