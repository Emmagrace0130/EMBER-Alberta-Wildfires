from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..services.rag_service import stream_chat

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []


@router.post("")
async def chat(req: ChatRequest) -> StreamingResponse:
    history = [{"role": t.role, "content": t.content} for t in req.history]
    return StreamingResponse(
        stream_chat(req.message, history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
