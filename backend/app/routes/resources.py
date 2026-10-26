from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..core.config import DOCUMENTS_DIR
from ..schemas import Resource
from ..services import content_service

router = APIRouter(prefix="/api/resources", tags=["resources"])


@router.get("", response_model=list[Resource])
def list_resources() -> list[Resource]:
    resources = content_service.get_resources()
    return [
        r.model_copy(update={"downloadUrl": f"/api/resources/{r.id}/download"})
        if r.fileName
        else r
        for r in resources
    ]


@router.get("/{resource_id}/download")
def download_resource(resource_id: str) -> FileResponse:
    resource = content_service.get_resource(resource_id)
    if resource is None or not resource.fileName:
        raise HTTPException(status_code=404, detail="Resource has no downloadable file")

    file_path = DOCUMENTS_DIR / resource.fileName
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Source file is unavailable on the server")

    return FileResponse(file_path, filename=resource.fileName, media_type="application/pdf")
