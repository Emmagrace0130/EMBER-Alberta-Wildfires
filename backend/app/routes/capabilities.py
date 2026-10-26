from fastapi import APIRouter

from ..schemas import Capability
from ..services import content_service

router = APIRouter(prefix="/api/capabilities", tags=["capabilities"])


@router.get("", response_model=list[Capability])
def list_capabilities() -> list[Capability]:
    return content_service.get_capabilities()
