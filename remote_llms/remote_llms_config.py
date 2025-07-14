from enum import Enum
from pathlib import Path
from typing import Any, List, Dict

from pydantic import (
    BaseModel,
    Field,
)

from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource, PydanticBaseSettingsSource


class EndpointPlatforms(Enum):
    BEDROCK = "bedrock"
    OPENAI = "openai"
    GOOGLE = "google"


class LlmModel(BaseModel):
    name_or_url: str | Dict[str, str]
    endpoint_platform: EndpointPlatforms


class Settings(BaseSettings):
    models: Dict[str, LlmModel] = Field(..., alias='models')
    languages: Dict[str, str] = Field(..., alias='languages')
    model_config = SettingsConfigDict(yaml_file=Path(__file__).parent.parent / "llm-models-config.yml")

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


class AwsKeys(BaseSettings):
    aws_region: str = Field(..., alias='AWS_REGION')
    aws_access_key_id: str = Field(..., alias='AWS_SERVER_PUBLIC_KEY')
    aws_secret_access_key: str = Field(..., alias='AWS_SERVER_SECRET_KEY')

settings = Settings()
# print(f"{settings.models = }, {settings.languages = }")
# print(Settings().model_dump())