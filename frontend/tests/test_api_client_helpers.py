"""The organization/discussion helpers send their query parameters as parameters and a numeric timeout (a second
helper of the same name once replaced the first, so list_comments passed its parameters as the timeout)."""
import api_client


class _Resp:
    status_code = 200
    ok = True
    text = "[]"

    def json(self):
        return []


def test_list_comments_and_audit_events_send_params_not_a_timeout(monkeypatch):
    seen = []
    monkeypatch.setattr(api_client, "_headers", lambda **k: {})
    monkeypatch.setattr(api_client.requests, "get", lambda url, **kw: seen.append((url, kw)) or _Resp())
    api_client.list_comments(status="open")
    api_client.org_audit_events(1, limit=5)
    url, kw = seen[0]
    assert url.endswith("/api/discussion") and kw["params"] == {"status": "open"}
    assert isinstance(kw["timeout"], (int, float))
    assert seen[1][1]["params"] == {"limit": 5}
    assert all(isinstance(k["timeout"], (int, float)) for _, k in seen)
