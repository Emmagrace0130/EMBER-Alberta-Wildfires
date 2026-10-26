import json
from functools import lru_cache

from ..core.config import DATA_DIR
from ..schemas import Capability, MitigationPlan, Resource, RiskInsights


def _load(filename: str) -> list | dict:
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


@lru_cache
def get_plans() -> list[MitigationPlan]:
    return [MitigationPlan(**p) for p in _load("plans.json")]


def get_plan(plan_id: str) -> MitigationPlan | None:
    return next((p for p in get_plans() if p.id == plan_id), None)


@lru_cache
def get_resources() -> list[Resource]:
    return [Resource(**r) for r in _load("resources.json")]


def get_resource(resource_id: str) -> Resource | None:
    return next((r for r in get_resources() if r.id == resource_id), None)


@lru_cache
def get_capabilities() -> list[Capability]:
    return [Capability(**c) for c in _load("capabilities.json")]


@lru_cache
def get_risk_insights() -> RiskInsights:
    return RiskInsights(**_load("risk.json"))


def search(query: str) -> list[dict]:
    q = query.strip().lower()
    if not q:
        return []

    results: list[dict] = []
    for plan in get_plans():
        haystack = f"{plan.title} {plan.summary} {' '.join(plan.categories)}".lower()
        if q in haystack:
            results.append(
                {
                    "id": plan.id,
                    "type": "plan",
                    "title": plan.title,
                    "snippet": plan.summary,
                }
            )
    for resource in get_resources():
        haystack = f"{resource.title} {resource.description} {resource.type}".lower()
        if q in haystack:
            results.append(
                {
                    "id": resource.id,
                    "type": "resource",
                    "title": resource.title,
                    "snippet": resource.description,
                }
            )
    return results
