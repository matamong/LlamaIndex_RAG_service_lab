from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = '/api/v1'
    DOMAIN: str = 'localhost'
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    # DB
    DB_NAME: str
    DB_ENDPOINT: str
    DB_PORT: str
    DB_USERNAME: str
    DB_PASSWORD: str

    # Logging
    LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERRO', 'CRITICAL'] = 'DEBUG'

settings = Settings()
