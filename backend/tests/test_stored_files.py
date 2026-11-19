"""An uploaded file survives the disk being wiped (a redeploy on Render's free tier)."""
from __future__ import annotations

import os

import pytest

from app import config, stored_files
from app.database import SessionLocal
from app.models import StoredFile

CSV = b"a,b\n1,2\n3,4\n"


pytestmark = pytest.mark.usefixtures("client")   # starting the app creates the tables


@pytest.fixture
def path():
    p = config.UPLOADS_DIR / "stored_files_test.csv"
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    p.write_bytes(CSV)
    yield str(p)
    p.unlink(missing_ok=True)
    db = SessionLocal()
    row = db.get(StoredFile, str(p))
    if row:
        db.delete(row)
        db.commit()
    db.close()


def test_a_saved_file_is_restored_after_the_disk_is_wiped(path):
    db = SessionLocal()
    assert stored_files.save(db, path)
    db.close()
    os.remove(path)
    assert stored_files.ensure(path)
    assert open(path, "rb").read() == CSV


def test_a_file_that_was_never_saved_is_reported_missing(path):
    os.remove(path)
    assert stored_files.ensure(path) is False


def test_nothing_is_written_outside_the_uploads_folder(tmp_path):
    outside = str(tmp_path / "evil.csv")
    db = SessionLocal()
    db.add(StoredFile(filepath=outside, content=b"x", size_bytes=1))
    db.commit()
    assert stored_files.ensure(outside) is False and not os.path.exists(outside)
    db.delete(db.get(StoredFile, outside))
    db.commit()
    db.close()


def test_files_over_the_limit_are_not_stored(path, monkeypatch):
    monkeypatch.setattr(config, "STORE_UPLOAD_BYTES", 3)
    db = SessionLocal()
    assert stored_files.save(db, path) is False
    db.close()


def test_upload_keeps_a_copy_and_startup_puts_the_file_back(client):
    from app.models import Dataset
    from tests.platform_helpers import workspace
    ws = workspace(client)
    db = SessionLocal()
    fp = db.get(Dataset, ws["ds"]).filepath
    assert db.get(StoredFile, fp) is not None, "the upload kept a copy in the database"
    db.close()
    os.remove(fp)
    assert stored_files.restore_missing() >= 1 and os.path.isfile(fp)


def test_an_analysis_of_a_file_that_is_gone_says_so_instead_of_an_internal_error(client):
    from app.models import Dataset, StoredFile as SF
    from tests.platform_helpers import ask_and_run, question_row, workspace
    ws = workspace(client)
    db = SessionLocal()
    fp = db.get(Dataset, ws["ds"]).filepath
    db.delete(db.get(SF, fp))
    db.commit()
    db.close()
    os.remove(fp)
    out = ask_and_run(client, ws, "What is the total revenue?")
    q = question_row(out["question_id"])
    assert q.status == "failed" and "upload the file again" in q.error and "internal error" not in q.error


def test_a_file_saved_under_another_machines_path_is_restored_here(path):
    """Dataset rows carry the uploader's absolute path (a Windows or /app path); resolve() finds it by file name."""
    name = os.path.basename(path)
    foreign = r"C:\Users\someone\backend\data\uploads" + "\\" + name
    db = SessionLocal()
    db.add(StoredFile(filepath=foreign, content=CSV, size_bytes=len(CSV)))
    db.commit()
    os.remove(path)
    try:
        got = stored_files.resolve(foreign)
        assert got is not None and os.path.samefile(got, path) and open(got, "rb").read() == CSV
        assert stored_files.resolve("/app/data/uploads/never_saved.csv") is None
    finally:
        db.delete(db.get(StoredFile, foreign))
        db.commit()
        db.close()
