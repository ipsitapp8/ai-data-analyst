"""Keep uploaded files in the database and put them back on disk when they are gone.

On a host whose disk is wiped at every deploy (Render's free tier), the dataset rows survive in
Postgres but the CSV they point at does not, so a later analysis would crash on a missing file.
add_version() saves each upload here; restore_missing() runs at startup and ensure() runs before an
analysis. A file is only ever written back inside UPLOADS_DIR, whatever the stored path says.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from app import config
from app.database import SessionLocal
from app.models import DatasetVersion, StoredFile

logger = logging.getLogger(__name__)


def _inside_uploads(path: str) -> bool:
    try:
        Path(path).resolve().relative_to(config.UPLOADS_DIR.resolve())
        return True
    except (ValueError, OSError):
        return False


def save(db, filepath: str) -> bool:
    """Store a copy of the file. Never raises: failing to back up must not fail the upload."""
    try:
        limit = config.STORE_UPLOAD_BYTES
        if limit <= 0 or not os.path.isfile(filepath) or os.path.getsize(filepath) > limit:
            return False
        with open(filepath, "rb") as f:
            data = f.read()
        db.merge(StoredFile(filepath=filepath, content=data, size_bytes=len(data)))
        db.commit()
        return True
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.warning("Could not store a copy of %s", filepath, exc_info=True)
        return False


def _write(filepath: str, content: bytes) -> bool:
    if not _inside_uploads(filepath):
        logger.warning("Not restoring %s: outside the uploads folder", filepath)
        return False
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".restoring")
    tmp.write_bytes(content)
    os.replace(tmp, dest)
    return True


def ensure(filepath: str) -> bool:
    """True if the file is on disk afterwards, restoring it from the database if it was lost."""
    if os.path.isfile(filepath):
        return True
    db = SessionLocal()
    try:
        row = db.get(StoredFile, filepath)
        if row is None:
            return False
        return _write(filepath, bytes(row.content))
    except Exception:  # noqa: BLE001
        logger.warning("Could not restore %s", filepath, exc_info=True)
        return False
    finally:
        db.close()


def resolve(filepath: str) -> str | None:
    """A path under UPLOADS_DIR that holds the file, or None if it exists nowhere.

    Dataset rows keep the absolute path of the machine that took the upload (C:\\Users\\... from a
    laptop, /app/data/... from Render), and several machines can share one database. So when the
    stored path is not usable here, look for the same file name in this machine's uploads folder
    and restore the saved copy there. Upload names start with a random id, so the name is unique.
    """
    if os.path.isfile(filepath):
        return filepath
    local = config.UPLOADS_DIR / filepath.replace("\\", "/").rsplit("/", 1)[-1]
    if local.is_file():
        return str(local)
    db = SessionLocal()
    try:
        row = db.get(StoredFile, filepath)
        if row is None:
            return None
        return str(local) if _write(str(local), bytes(row.content)) else None
    except Exception:  # noqa: BLE001
        logger.warning("Could not restore %s", filepath, exc_info=True)
        return None
    finally:
        db.close()


def restore_missing() -> int:
    """At startup: put back every stored file a dataset version points at and the disk lacks."""
    restored = 0
    db = SessionLocal()
    try:
        paths = {p for (p,) in db.query(DatasetVersion.filepath).distinct()}
        for path in paths:
            if not os.path.isfile(path) and ensure(path):
                restored += 1
    except Exception:  # noqa: BLE001
        logger.warning("Restoring uploaded files failed", exc_info=True)
    finally:
        db.close()
    if restored:
        logger.info("Restored %d uploaded file(s) from the database", restored)
    return restored


def available(db, filepath: str) -> bool:
    """True if the file can be read here, or restored: on disk (at its stored path or in this uploads folder)
    or saved in the database. Used to hide datasets whose file is gone for good."""
    if os.path.isfile(filepath) or (config.UPLOADS_DIR / filepath.replace("\\", "/").rsplit("/", 1)[-1]).is_file():
        return True
    try:
        return db.get(StoredFile, filepath) is not None
    except Exception:  # noqa: BLE001
        db.rollback()
        return True   # not knowing is not a reason to hide a dataset
