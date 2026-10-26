import json
from typing import AsyncGenerator

import chromadb
import httpx

from ..core.config import OLLAMA_URL, VECTOR_STORE_DIR

EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "llama3.1:8b"
COLLECTION_NAME = "wildfire_docs"
RETRIEVE_K = 5

SYSTEM_PROMPT = (
    "You are EMBER, a wildfire mitigation and decision-support assistant for Alberta, Canada. "
    "You help communities, planners, emergency managers, and researchers understand wildfire risk "
    "and mitigation strategies. Answer questions using the provided context from EMBER's document "
    "library. Be specific, practical, and actionable. If the context does not contain enough "
    "information to answer the question fully, say so clearly rather than making up facts. "
    "Keep responses concise and well-structured."
)

_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _friendly_title(raw: str) -> str:
    mapping = {
        "7032001-2013-11-firesmart-guidebook-community-protection-guidebook-wildland-urba": "FireSmart Community Protection Guidebook",
        "af-farm-and-acreage-guide-to-reducing-wildfire-risk-2020-08-25": "Farm & Acreage Wildfire Risk Reduction Guide",
        "alberta-wildland-urban-interface-fires": "Alberta WUI Fires Analysis",
        "CWFIS_DataServices_HowtoAccessDailyMaps&DataLayers": "CWFIS Data Services Guide",
        "p-310-09-ho1-prevention-plan-template": "BLM Wildfire Prevention Plan Template",
        "rmrs_gtr292": "USFS RMRS-GTR-292 Wildfire Research",
    }
    return mapping.get(raw, raw)


async def _embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def stream_chat(message: str, history: list[dict]) -> AsyncGenerator[str, None]:
    # Embed and retrieve context
    try:
        embedding = await _embed(message)
        col = _get_collection()
        results = col.query(
            query_embeddings=[embedding],
            n_results=RETRIEVE_K,
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []
    sources: list[dict] = []
    seen: set[str] = set()

    for i, (chunk, meta) in enumerate(zip(chunks, metas)):
        raw_title = meta.get("doc_title", "Unknown")
        friendly = _friendly_title(raw_title)
        context_parts.append(f"[Source {i + 1}: {friendly}]\n{chunk}")
        if raw_title not in seen:
            seen.add(raw_title)
            sources.append({"title": friendly, "raw": raw_title})

    context = "\n\n---\n\n".join(context_parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"Context from EMBER's document library:\n\n{context}\n\nQuestion: {message}",
    })

    # Stream from Ollama
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json={"model": CHAT_MODEL, "messages": messages, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"
                    if data.get("done"):
                        break
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'content': f'Generation error: {exc}'})}\n\n"

    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
    yield "data: [DONE]\n\n"
