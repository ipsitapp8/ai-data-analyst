"""Types shared by every domain pack. A domain describes; the shared engine calculates."""
from __future__ import annotations

from dataclasses import dataclass, field

# feasibility of a suggested question
READY = "ready"
NEEDS_ASSUMPTIONS = "needs_assumptions"
NEEDS_EXTERNAL_DATA = "needs_external_data"
UNSUPPORTED = "unsupported"
FEASIBILITY = (READY, NEEDS_ASSUMPTIONS, NEEDS_EXTERNAL_DATA, UNSUPPORTED)

# what kind of statement a result is
EXACT, ESTIMATE, FORECAST, HYPOTHETICAL = "exact_calculation", "statistical_estimate", "forecast", "hypothetical_assumption"


@dataclass(frozen=True)
class Concept:
    """An idea a column may express ("attendance"). `tokens` are lower-case words of a column name that suggest it."""
    key: str
    label: str
    tokens: frozenset
    kind: str = "numeric"           # numeric | category | id
    weight: float = 1.0             # how much a match counts as evidence for the domain
    multi: bool = False             # several columns may express it (one per subject)


@dataclass(frozen=True)
class InputSpec:
    """One value a template needs from the user. `default` None means the user must supply it (never invented)."""
    name: str
    label: str
    unit: str = ""
    default: float | None = None
    min: float | None = None
    max: float | None = None
    from_data: str | None = None    # a derivation the engine can make from the dataset, e.g. "entity_count"
    help: str = ""
    role: str = "value"             # value (an operation's own parameter) | baseline | change | assumption
    target: str | None = None       # for role "change": the model input it adjusts
    change_type: str = "pct"        # for role "change": pct | abs | set
    optional: bool = False


@dataclass(frozen=True)
class Template:
    id: str
    title: str
    question: str
    method: str                     # deterministic | monte_carlo | sensitivity | break_even | goal_seek | forecast
    operation: str                  # key in whatif.operations.OPERATIONS, or "formula_model"
    requires: tuple = ()            # concept keys that must map to columns in the dataset
    inputs: tuple = ()              # InputSpec the user supplies
    reason: str = ""
    tags: frozenset = frozenset()   # words used to match an objective or a typed question
    limitation: str = ""
    evidence: str = EXACT
    manual: bool = False            # can run on typed values alone, no dataset needed
    external: tuple = ()            # information no dataset holds (named, so the user knows what is missing)
    params: dict = field(default_factory=dict)   # fixed parameters for the operation (which concept, which direction)
    model: dict | None = None       # formula-model definition (see app.scenario_models) when operation == "formula_model"
    patterns: tuple = ()            # regexes a typed question is matched against (Copilot and free-text)
    unsupported: str = ""           # when set, why no calculation here can answer it reliably


@dataclass(frozen=True)
class Domain:
    key: str
    title: str
    description: str
    entities: tuple
    concepts: tuple
    templates: tuple
    metrics: tuple = ()
    keywords: frozenset = frozenset()   # extra words that point at the domain (matched against column names)
    sensitive_tokens: frozenset = frozenset()   # columns never echoed back in outputs
    notes: tuple = ()               # domain-specific explanation / safety notes shown with results

    def template(self, template_id: str) -> Template | None:
        return next((t for t in self.templates if t.id == template_id), None)

    def concept(self, key: str) -> Concept | None:
        return next((c for c in self.concepts if c.key == key), None)
