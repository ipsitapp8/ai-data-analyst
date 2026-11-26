"""Uploading the same file twice must not create a second dataset."""
from __future__ import annotations

from tests.platform_helpers import SALES_CSV, DRIFTED_CSV, workspace


def _upload(client, ws, name, data):
    return client.post("/api/datasets/upload", files={"file": (name, data, "text/csv")}, headers=ws["h"])


def test_same_file_is_not_added_twice(client):
    ws = workspace(client)
    again = _upload(client, ws, "sales.csv", SALES_CSV).json()
    assert again["id"] == ws["ds"] and again["already_existed"] is True
    listed = client.get("/api/datasets", headers=ws["h"]).json()
    assert len(listed) == 1


def test_changed_content_or_new_name_is_a_new_dataset(client):
    ws = workspace(client)
    assert _upload(client, ws, "sales.csv", DRIFTED_CSV).json()["already_existed"] is False
    assert _upload(client, ws, "sales_copy.csv", SALES_CSV).json()["already_existed"] is False
    assert len(client.get("/api/datasets", headers=ws["h"]).json()) == 3
