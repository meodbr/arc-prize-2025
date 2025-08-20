from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ValidationError
import os
import sys


class Settings(BaseSettings):
    GOOGLE_CLOUD_PROJECT: str = Field(..., description="ID de projet GCP")
    GOODLE_CLOUD_REGION: str = Field(..., description="Région GCP")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

try:
    settings = Settings()
except ValidationError as e:
    if not os.path.exists(".env"):
        print(
            ".env file not found! Please create one or copy from .env.example.",
            file=sys.stderr,
        )
    else:
        print("Configuration error:", file=sys.stderr)
    print(e, file=sys.stderr)
    sys.exit(1)
