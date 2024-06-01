from fastapi import FastAPI

from app.api.api_router import api_router
from app.utils.logging import AppLogger


logger = AppLogger().get_logger()


app = FastAPI()

app.include_router(api_router, prefix='/api/v1')

logger.info('[Main] FastAPI app started.')