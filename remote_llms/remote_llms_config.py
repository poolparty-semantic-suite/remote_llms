import logging
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict

from pydantic import (
    BaseModel,
    Field,
)

from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource, PydanticBaseSettingsSource

from remote_llms import clients


class EndpointPlatforms(Enum):
    BEDROCK = "bedrock"
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"


class LlmModel(BaseModel):
    name_or_url: str | Dict[str, str]
    endpoint_platform: EndpointPlatforms

    def get_uid(self) -> str:
        if isinstance(self.name_or_url, str):
            return self.name_or_url
        else:
            aws_settings = AwsKeys()
            region = aws_settings.aws_region[:2]
            logging.info(f"AWS_REGION: {region}")
            return self.name_or_url[region]


class Settings(BaseSettings):
    models: Dict[str, LlmModel] = Field(..., alias='models')
    languages: Dict[str, str] = Field(..., alias='languages')
    model_config = SettingsConfigDict(yaml_file=Path(__file__).parent / "llm-models-config.yml")

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

    def model_post_init(self, context: Any, /) -> None:
        def filter_models(client, platform: EndpointPlatforms) -> dict[str, LlmModel]:
            try:
                client()
            except EnvironmentError:
                filtered_models = {}
                for model_name, model in self.models.items():
                    if model.endpoint_platform == platform:
                        continue
                    else:
                        filtered_models[model_name] = model
                self.models = filtered_models
            return self.models

        for cl, pl in [(clients.bedrock_client, EndpointPlatforms.BEDROCK),
                       (clients.openai_client, EndpointPlatforms.OPENAI),
                       (clients.google_client, EndpointPlatforms.GOOGLE),
                       (clients.ollama_client, EndpointPlatforms.OLLAMA),]:
            filter_models(cl, pl)


class AwsKeys(BaseSettings):
    aws_region: str = Field(..., alias='AWS_REGION')
    aws_access_key_id: str = Field(..., alias='AWS_SERVER_PUBLIC_KEY')
    aws_secret_access_key: str = Field(..., alias='AWS_SERVER_SECRET_KEY')

settings = Settings()
# print(f"{settings.models = }, {settings.languages = }")
# print(Settings().model_dump())