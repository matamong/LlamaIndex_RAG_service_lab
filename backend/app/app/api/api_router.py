from fastapi import APIRouter

from app.api.endpoints import endpoint_ping

api_router = APIRouter()

api_router.include_router(endpoint_ping.router, tags=["ping"])