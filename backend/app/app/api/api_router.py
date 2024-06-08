from fastapi import APIRouter

from app.api.endpoints import endpoint_ping, endpoint_infer

api_router = APIRouter()

api_router.include_router(endpoint_ping.router, tags=["ping"])
api_router.include_router(endpoint_infer.router, tags=["infer"])