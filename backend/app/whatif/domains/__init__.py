"""The domain registry. Each domain pack describes a sector; the shared engine runs what it describes."""
from __future__ import annotations

from .base import Domain, Template
from .business import BUSINESS
from .custom import CUSTOM, GENERIC_TEMPLATES
from .education import EDUCATION
from .government import GOVERNMENT
from .healthcare import HEALTHCARE
from .travel import TRAVEL

_REGISTRY: dict[str, Domain] = {}


def register_domain(domain: Domain) -> None:
    if domain.key in _REGISTRY:
        raise ValueError(f"Domain '{domain.key}' is already registered")
    _REGISTRY[domain.key] = domain


def get_domain(key: str) -> Domain:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"Unknown domain '{key}'. Available: {', '.join(_REGISTRY)}") from None


def list_domains() -> list[Domain]:
    return list(_REGISTRY.values())


def find_template(domain_key: str, template_id: str) -> Template | None:
    if template_id in GENERIC_TEMPLATES:
        return GENERIC_TEMPLATES[template_id]
    return get_domain(domain_key).template(template_id)


for _d in (BUSINESS, EDUCATION, HEALTHCARE, TRAVEL, GOVERNMENT, CUSTOM):
    register_domain(_d)

# detect_domain lives in whatif.profile because it needs a dataset profile; it reads this registry.
