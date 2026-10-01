"""Dataset versioning helpers.

A Dataset is a stable "slot" (its id never changes); each upload into it is a
DatasetVersion. The Dataset row mirrors the latest version's file, counts and
profile so code that only knows about Dataset keeps working.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Dataset, DatasetVersion


def add_version(
    db: Session, dataset: Dataset, *, filename: str, filepath: str, profile: dict
) -> DatasetVersion:
    """Append a new version and point the dataset's mirror fields at it."""
    current = latest_version(db, dataset)
    version = DatasetVersion(
        dataset_id=dataset.id,
        version_number=(current.version_number + 1) if current else 1,
        filename=filename,
        filepath=filepath,
        row_count=profile["row_count"],
        col_count=profile["col_count"],
        profile_json=profile,
    )
    db.add(version)
    dataset.filename = filename
    dataset.filepath = filepath
    dataset.row_count = profile["row_count"]
    dataset.col_count = profile["col_count"]
    dataset.profile_json = profile
    db.commit()
    db.refresh(version)
    db.refresh(dataset)
    return version


def latest_version(db: Session, dataset: Dataset) -> DatasetVersion | None:
    return (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset.id)
        .order_by(DatasetVersion.version_number.desc())
        .first()
    )


def version_count(db: Session, dataset_id: int) -> int:
    return db.query(DatasetVersion).filter(DatasetVersion.dataset_id == dataset_id).count()


def backfill_dataset_versions() -> None:
    """Give every pre-versioning dataset a version 1 built from its own row.
    Idempotent: datasets that already have a version are skipped."""
    db = SessionLocal()
    try:
        have = {row[0] for row in db.query(DatasetVersion.dataset_id).distinct()}
        for d in db.query(Dataset).all():
            if d.id in have:
                continue
            db.add(DatasetVersion(
                dataset_id=d.id, version_number=1, filename=d.filename, filepath=d.filepath,
                row_count=d.row_count, col_count=d.col_count, profile_json=d.profile_json,
                uploaded_at=d.uploaded_at,
            ))
        db.commit()
    finally:
        db.close()
