from fastapi import APIRouter

from ..schemas import RiskInsights
from ..services import content_service

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("", response_model=RiskInsights)
def get_risk_insights() -> RiskInsights:
    return content_service.get_risk_insights()
