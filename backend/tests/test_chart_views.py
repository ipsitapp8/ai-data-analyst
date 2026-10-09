"""Chart Studio view overrides (team-scoped) and CSV line tracing for Inspect."""
from __future__ import annotations

import json
import uuid

from app.csv_lines import match_csv_lines
from app.database import SessionLocal
from app.models import Dashboard, Question


def _signup(client, email: str) -> str:
    r = client.post("/api/auth/signup", json={"email": email, "password": "password123", "display_name": "T"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _team_headers(client, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    community = client.post("/api/communities", json={"name": "Co"}, headers=headers).json()
    team = client.post(f"/api/communities/{community['id']}/teams", json={"name": "T"}, headers=headers).json()
    return {**headers, "X-Team-Id": str(team["id"])}


def _seed_dashboard(client, headers: dict) -> tuple[int, int, str]:
    files = {"file": ("t.csv", b"a,b\n1,2\n", "text/csv")}
    dataset_id = client.post("/api/datasets/upload", files=files, headers=headers).json()["id"]
    element_id = uuid.uuid4().hex
    db = SessionLocal()
    try:
        team_id = int(headers["X-Team-Id"])
        q = Question(team_id=team_id, dataset_id=dataset_id, text="q", status="verified")
        db.add(q)
        db.commit()
        d = Dashboard(team_id=team_id, question_id=q.id,
                      charts_json=[{"title": "c", "plotly_json": {}, "element_id": element_id}])
        db.add(d)
        db.commit()
        return q.id, d.id, element_id
    finally:
        db.close()


def test_save_read_and_reset_a_chart_view(client, unique_email):
    headers = _team_headers(client, _signup(client, unique_email))
    question_id, dashboard_id, element_id = _seed_dashboard(client, headers)
    url = f"/api/dashboards/{dashboard_id}/charts/{element_id}/view"

    r = client.put(url, json={"chart_type": "donut", "style": {"effect": "glass"}}, headers=headers)
    assert r.status_code == 200, r.text
    views = client.get(f"/api/questions/{question_id}/dashboard", headers=headers).json()["view_overrides"]
    assert views[element_id] == {"chart_type": "donut", "style": {"effect": "glass"}}

    assert client.delete(url, headers=headers).status_code == 200
    assert client.get(f"/api/questions/{question_id}/dashboard", headers=headers).json()["view_overrides"] == {}


def test_chart_view_rejects_unknown_chart_and_bad_type(client, unique_email):
    headers = _team_headers(client, _signup(client, unique_email))
    _, dashboard_id, element_id = _seed_dashboard(client, headers)
    assert client.put(f"/api/dashboards/{dashboard_id}/charts/nope/view",
                      json={"chart_type": "line"}, headers=headers).status_code == 404
    assert client.put(f"/api/dashboards/{dashboard_id}/charts/{element_id}/view",
                      json={"chart_type": "<script>"}, headers=headers).status_code == 422


def test_chart_view_is_team_isolated(client, unique_email):
    alice = _team_headers(client, _signup(client, f"a-{unique_email}"))
    bob = _team_headers(client, _signup(client, f"b-{unique_email}"))
    _, dashboard_id, element_id = _seed_dashboard(client, alice)
    r = client.put(f"/api/dashboards/{dashboard_id}/charts/{element_id}/view",
                   json={"chart_type": "pie"}, headers=bob)
    assert r.status_code == 404


def test_match_csv_lines(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("id,date,amount,region\n1,2025-01-05,1500,North\n2,2025-01-06,99.5,South\n"
                    "3,2025-01-05,1500,North\n")
    rows = [[2, "2025-01-06 00:00:00", 99.5, "South"],
            [1, "2025-01-05", 1500.0, "North"],
            [9, "2025-01-05", 1, "Nowhere"]]
    assert match_csv_lines(str(path), ["id", "date", "amount", "region"], rows) == [3, 2, None]
    # duplicate rows map to successive lines, not the same one twice
    dupes = [["2025-01-05", 1500], ["2025-01-05", 1500]]
    assert match_csv_lines(str(path), ["date", "amount"], dupes) == [2, 4]
    # a derived column means aggregated rows: no line numbers
    assert match_csv_lines(str(path), ["region", "total"], [["North", 3000]]) == [None]
