# `remote_llms`

Package to provide access to remote LLMs. The main idea of this package is to keep model control centralized. This might help mitigate problems with model versioning and provide an easy access to many LLMs with very low effort.

The list of supported LLMs can be found in (./llm-models-config.yml). WARNING: currently only bedrock are checked to be working.

Main usage: `call_llm` from (./remote_llms.py)

## Installation

`[poetry/uv] add git+https://github.com/poolparty-semantic-suite/remote_llms.git`

## Usage in your endpoints

To add a choice of LLMs to your endpoints you can define the respective `enum`:
`RemoteLlmsEnum = Enum("RemoteLlmsEnum", {x: x for x in remote_llms.remote_llms_settings.models.keys()})`
and then in your models add a field:
`llm_model: RemoteLlmsEnum = Field(default=list(remote_llms.remote_llms_settings.models.keys())[0], alias="llmModel")`

## Env variables / Config

The following variables are expected (per platform):
  * bedrock keys
     * AWS_REGION=eu-central-1
     * AWS_SERVER_PUBLIC_KEY=
     * AWS_SERVER_SECRET_KEY=
  * open ai keys
     * OPENAI_API_KEY=
     * OPENAI_ORG_ID=
     * OPENAI_PROJECT_ID=
  * gemini keys
     * GEMINI_API_KEY=

## Updating models

To add a model:
1. add the model data to [./llm-models-config.yml](remote_llms/llm-models-config.yml).
2. add a function to call the model to [./remote_llms.py](./remote_llms.py).
3. update the function `call_llm` in [./remote_llms.py](./remote_llms.py).

To add a new platform:
1. add a client for platform. see, for example, `bedrock_client` in [./remote_llms.py](./remote_llms.py).
2. update the function `call_llm` in [./remote_llms.py](./remote_llms.py).
3. add new env variables to the `Settings` in [./remote_llms_config.py](./remote_llms_config.py) and to the env example.