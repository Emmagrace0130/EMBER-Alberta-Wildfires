from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .core.config import CORS_ORIGINS
from .routes import capabilities, chat, contact, health, map, plans, resources, risk, search

app = FastAPI(title="EMBER Wildfire Mitigation Platform API", version="0.1.0")

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(plans.router)
app.include_router(risk.router)
app.include_router(capabilities.router)
app.include_router(resources.router)
app.include_router(search.router)
app.include_router(contact.router)
app.include_router(map.router)
app.include_router(chat.router)
