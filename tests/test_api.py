def test_full_run_lifecycle(client, auth_headers):
    resp = client.post("/runs", json={"topic": "Agentic AI"}, headers=auth_headers)
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    resp = client.get(f"/runs/{run_id}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "awaiting_approval"
    assert data["blog_title"]
    assert data["blog_content"]

    resp = client.post(f"/runs/{run_id}/decision", json={"decision": "approved"}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"

    resp = client.get(f"/runs/{run_id}", headers=auth_headers)
    assert resp.json()["status"] == "approved"


def test_rejected_decision(client, auth_headers):
    resp = client.post("/runs", json={"topic": "Agentic AI"}, headers=auth_headers)
    run_id = resp.json()["run_id"]

    resp = client.post(f"/runs/{run_id}/decision", json={"decision": "rejected"}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"


def test_start_run_requires_topic(client, auth_headers):
    resp = client.post("/runs", json={}, headers=auth_headers)
    assert resp.status_code == 400


def test_requires_api_key(client):
    resp = client.post("/runs", json={"topic": "Agentic AI"})
    assert resp.status_code == 401


def test_get_unknown_run_returns_404(client, auth_headers):
    resp = client.get("/runs/does-not-exist", headers=auth_headers)
    assert resp.status_code == 404


def test_decision_on_already_decided_run_returns_409(client, auth_headers):
    resp = client.post("/runs", json={"topic": "Agentic AI"}, headers=auth_headers)
    run_id = resp.json()["run_id"]

    client.post(f"/runs/{run_id}/decision", json={"decision": "approved"}, headers=auth_headers)
    resp = client.post(f"/runs/{run_id}/decision", json={"decision": "approved"}, headers=auth_headers)
    assert resp.status_code == 409
