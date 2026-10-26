from fastapi import APIRouter, HTTPException

from ..schemas import MitigationPlan
from ..services import content_service

router = APIRouter(prefix="/api/plans", tags=["plans"])


@router.get("", response_model=list[MitigationPlan])
def list_plans() -> list[MitigationPlan]:
    plans = content_service.get_plans()
    return [
        plan.model_copy(update={"downloadUrl": f"/api/resources/{plan.id}/download"})
        for plan in plans
    ]


@router.get("/{plan_id}", response_model=MitigationPlan)
def get_plan(plan_id: str) -> MitigationPlan:
    plan = content_service.get_plan(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Mitigation plan not found")
    return plan.model_copy(update={"downloadUrl": f"/api/resources/{plan.id}/download"})
