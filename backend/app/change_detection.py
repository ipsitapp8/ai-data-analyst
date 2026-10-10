"""KPI diffing for scheduled re-runs.

KPI values are display strings written by the LLM ("$1,234.5", "12.4%",
"1.2M"), so comparison starts by parsing a number out of them. KPIs are
matched across runs by normalised label; a KPI present in only one of the two
runs is ignored (label drift between LLM runs would otherwise page people).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}
FLAT_BELOW_PCT = 0.5  # |change| under this reads as "flat" in the trend arrow


def parse_kpi_value(text: str | None) -> float | None:
    if not text:
        return None
    m = _NUMBER.search(str(text))
    if not m:
        return None
    try:
        value = float(m.group().replace(",", ""))
    except ValueError:
        return None
    rest = str(text)[m.end():].lstrip()
    # A magnitude suffix only counts when it stands alone ("1.2M", "3 k"),
    # not when it starts a word ("12 months").
    if rest and rest[0].lower() in _SUFFIX and not rest[1:2].isalpha():
        value *= _SUFFIX[rest[0].lower()]
    return value


def _norm(label: str | None) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().lower())


@dataclass
class KpiChange:
    label: str
    old: str
    new: str
    pct: float | None  # None: not computable (old value was 0, or non-numeric)
    crossed: bool

    def describe(self) -> str:
        if self.pct is None:
            return f"{self.label}: {self.old} -> {self.new}"
        return f"{self.label}: {self.old} -> {self.new} ({self.pct:+.1f}%)"


def diff_kpis(old_kpis: list[dict], new_kpis: list[dict], threshold_pct: float) -> list[KpiChange]:
    """One KpiChange per KPI whose value differs between runs (crossed=True when
    it moved by more than the threshold)."""
    old_by_label = {_norm(k.get("label")): k for k in old_kpis or []}
    changes: list[KpiChange] = []
    for k in new_kpis or []:
        prev = old_by_label.get(_norm(k.get("label")))
        if prev is None:
            continue
        old_s, new_s = str(prev.get("value", "")), str(k.get("value", ""))
        if old_s == new_s:
            continue
        old_n, new_n = parse_kpi_value(old_s), parse_kpi_value(new_s)
        label = k.get("label", "")
        if old_n is None or new_n is None:
            # Text KPI that changed: no percentage to compare, so any change counts.
            changes.append(KpiChange(label, old_s, new_s, None, True))
        elif old_n == new_n:
            continue
        elif old_n == 0:
            changes.append(KpiChange(label, old_s, new_s, None, True))
        else:
            pct = (new_n - old_n) / abs(old_n) * 100
            changes.append(KpiChange(label, old_s, new_s, pct, abs(pct) > threshold_pct))
    return changes


def trend_of(changes: list[KpiChange]) -> str:
    """up/down/flat for the KPI that moved the most."""
    numeric = [c for c in changes if c.pct is not None]
    if not numeric:
        return "flat" if not changes else "up"
    biggest = max(numeric, key=lambda c: abs(c.pct))
    if abs(biggest.pct) < FLAT_BELOW_PCT:
        return "flat"
    return "up" if biggest.pct > 0 else "down"


@dataclass
class Anomaly:
    label: str
    value: str
    mean: float
    z: float | None  # None when history was perfectly flat
    runs: int

    def describe(self) -> str:
        if self.z is None:
            return f"{self.label}: {self.value} breaks a flat history of {self.mean:g} over {self.runs} runs"
        return (f"{self.label}: {self.value} is {abs(self.z):.1f} standard deviations "
                f"{'above' if self.z > 0 else 'below'} its usual {self.mean:g} over {self.runs} runs")


def detect_anomalies(history: list[list[dict]], new_kpis: list[dict], *,
                     z_threshold: float = 3.0, min_history: int = 5,
                     flat_tolerance: float = 0.05) -> list[Anomaly]:
    """KPIs whose new value is out of line with their own history.

    `history` is the KPI list of each earlier run. Unlike diff_kpis (which only
    compares against the previous run), this catches a value that is unusual
    even if it moved little since last time, and ignores a big move on a KPI
    that is normally volatile. Needs `min_history` numeric points per KPI, so a
    young schedule never fires. A perfectly flat history has no spread to take a
    z-score of, so it flags only a relative change over `flat_tolerance`.
    """
    import statistics

    past: dict[str, list[float]] = {}
    for kpis in history:
        for k in kpis or []:
            n = parse_kpi_value(str(k.get("value", "")))
            if n is not None:
                past.setdefault(_norm(k.get("label")), []).append(n)

    found: list[Anomaly] = []
    for k in new_kpis or []:
        values = past.get(_norm(k.get("label")), [])
        new = parse_kpi_value(str(k.get("value", "")))
        if new is None or len(values) < min_history:
            continue
        mean = statistics.fmean(values)
        sd = statistics.stdev(values)
        if sd == 0:
            if abs(new - mean) / (abs(mean) or 1.0) > flat_tolerance:
                found.append(Anomaly(k.get("label", ""), str(k.get("value")), mean, None, len(values)))
            continue
        z = (new - mean) / sd
        if abs(z) >= z_threshold:
            found.append(Anomaly(k.get("label", ""), str(k.get("value")), mean, z, len(values)))
    return found
