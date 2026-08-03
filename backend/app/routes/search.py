from fastapi import APIRouter, Query

from ..schemas import SearchResponse, SearchResultItem
from ..services import content_service

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search(q: str = Query(default="", max_length=200)) -> SearchResponse:
    results = content_service.search(q)
    return SearchResponse(
        query=q,
        results=[SearchResultItem(**r) for r in results],
    )
