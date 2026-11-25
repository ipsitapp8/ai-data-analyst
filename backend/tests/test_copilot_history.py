"""Saved Copilot conversations: persistence, search, rename, delete and per-user isolation."""
from __future__ import annotations

from tests.platform_helpers import add_member, workspace

BASE = "/api/copilot/conversations"


def _save(client, ws, cid, question, answer, dataset_id=None, position=0):
    h = ws["h"]
    assert client.put(f"{BASE}/{cid}", json={"dataset_id": dataset_id or ws["ds"]}, headers=h).status_code == 200
    r = client.put(f"{BASE}/{cid}/messages/{position}", headers=h,
                   json={"question": question, "response": {"answer": answer}, "answer_text": answer})
    assert r.status_code == 200, r.text


def test_conversation_round_trip_restores_messages_and_context(client):
    ws = workspace(client)
    _save(client, ws, "conv-one-aaaa", "Which region sells most?", "West sells the most, 42 percent.")
    _save(client, ws, "conv-one-aaaa", "And the least?", "North sells the least.", position=1)
    got = client.get(f"{BASE}/conv-one-aaaa", headers=ws["h"]).json()
    assert got["title"] == "Which region sells most?"            # the first question names the chat
    assert got["dataset_id"] == ws["ds"] and got["dataset_name"] == "sales.csv"
    assert [m["question"] for m in got["messages"]] == ["Which region sells most?", "And the least?"]
    assert got["messages"][0]["response"]["answer"].startswith("West")
    listed = client.get(BASE, headers=ws["h"]).json()["conversations"]
    assert [c["id"] for c in listed] == ["conv-one-aaaa"] and listed[0]["message_count"] == 2


def test_saving_the_same_message_twice_does_not_duplicate_it(client):
    ws = workspace(client)
    _save(client, ws, "conv-two-bbbb", "How many rows?", "720")
    _save(client, ws, "conv-two-bbbb", "How many rows?", "720 rows")
    msgs = client.get(f"{BASE}/conv-two-bbbb", headers=ws["h"]).json()["messages"]
    assert len(msgs) == 1 and msgs[0]["response"]["answer"] == "720 rows"


def test_search_matches_title_question_answer_and_dataset_name(client):
    ws = workspace(client)
    _save(client, ws, "conv-sea-aaaa", "Which region sells most?", "The West leads with strong revenue.")
    _save(client, ws, "conv-sea-bbbb", "List the products", "Widgets, gadgets and sprockets.")
    h = ws["h"]

    def ids(q):
        return {c["id"]: c for c in client.get(BASE, params={"q": q}, headers=h).json()["conversations"]}

    assert set(ids("region")) == {"conv-sea-aaaa"} and ids("region")["conv-sea-aaaa"]["matched_in"] == "title"
    hit = ids("sprockets")
    assert set(hit) == {"conv-sea-bbbb"} and hit["conv-sea-bbbb"]["matched_in"] == "answer"
    assert "sprockets" in hit["conv-sea-bbbb"]["snippet"]
    assert set(ids("sales.csv")) == {"conv-sea-aaaa", "conv-sea-bbbb"}      # dataset name
    assert ids("no such text anywhere") == {}
    assert ids("100%") == {}                                               # wildcards are escaped, not interpreted


def test_rename_and_delete(client):
    ws = workspace(client)
    _save(client, ws, "conv-ren-aaaa", "Hello there", "Hi")
    r = client.patch(f"{BASE}/conv-ren-aaaa", json={"title": "Quarterly review"}, headers=ws["h"])
    assert r.status_code == 200 and r.json()["title"] == "Quarterly review"
    assert client.delete(f"{BASE}/conv-ren-aaaa", headers=ws["h"]).status_code == 204
    assert client.get(f"{BASE}/conv-ren-aaaa", headers=ws["h"]).status_code == 404
    assert client.get(BASE, headers=ws["h"]).json()["conversations"] == []


def test_other_users_cannot_read_search_rename_or_delete_a_conversation(client):
    ws = workspace(client)
    _save(client, ws, "conv-priv-aaaa", "Confidential salary question", "The payroll is 9 million.")
    mate = add_member(client, ws)                       # a teammate in the SAME team
    assert client.get(BASE, headers=mate).json()["conversations"] == []
    assert client.get(BASE, params={"q": "payroll"}, headers=mate).json()["conversations"] == []
    assert client.get(f"{BASE}/conv-priv-aaaa", headers=mate).status_code == 404
    assert client.patch(f"{BASE}/conv-priv-aaaa", json={"title": "x"}, headers=mate).status_code == 404
    assert client.delete(f"{BASE}/conv-priv-aaaa", headers=mate).status_code == 404
    assert client.put(f"{BASE}/conv-priv-aaaa/messages/0", json={"question": "hijack"}, headers=mate).status_code == 404
    assert client.put(f"{BASE}/conv-priv-aaaa", json={}, headers=mate).status_code == 404
    still = client.get(f"{BASE}/conv-priv-aaaa", headers=ws["h"]).json()
    assert still["title"] == "Confidential salary question"


def test_login_is_required_and_bad_ids_are_refused(client):
    assert client.get(BASE).status_code in (401, 403)
    ws = workspace(client)
    assert client.put(f"{BASE}/a b", json={}, headers=ws["h"]).status_code in (400, 404, 422)
