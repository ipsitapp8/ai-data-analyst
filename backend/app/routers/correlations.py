"""Correlation explorer endpoint (see app/correlations.py)."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.correlations import top_correlations
from app.database import get_db
from app.dataset_versions import latest_version
from app.models import Dataset, Team, User
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/datasets/{dataset_id}/correlations", tags=["correlations"])


@router.get("")
def get_correlations(
    dataset_id: int,
    min_abs: float = 0.5,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    ds = db.get(Dataset, dataset_id)
    if not ds or ds.team_id != team.id:
        raise HTTPException(404, "Dataset not found")
    version = latest_version(db, ds)
    path = version.filepath if version else ds.filepath
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, "Could not read the dataset file") from exc
    return {"pairs": top_correlations(df, min_abs=min(max(min_abs, 0.0), 1.0))}
