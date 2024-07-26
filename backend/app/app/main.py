from contextlib import asynccontextmanager
from fastapi import FastAPI

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from app.api.api_router import api_router
from app.utils.logging import AppLogger
from app.config import settings


logger = AppLogger().get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('[Main] Setting up the app...')
    # set up the LLM and Embedding models if the OLLAMA_API_HOST is set
    logger.info('[Main] Set LLM and Embedding models.')
    if settings.OLLAMA_API_HOST:
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        Settings.llm = Ollama(base_url=settings.OLLAMA_URL, model="llama3", request_timeout=10000.0)
    logger.info('[Main] App setup complete.')
    # set up the gpt version if the OPENAI_API_KEY is set
    if settings.OPENAI_API_KEY:
        Settings.llm = OpenAI(model="gpt-4o")
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix='/api/v1')

logger.info('[Main] FastAPI app started.')