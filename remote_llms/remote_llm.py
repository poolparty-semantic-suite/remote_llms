import http
import json
import logging
import os

import boto3
# import requests
import ollama
from google import genai
from google.genai import types
from openai import OpenAI

from .remote_llms_config import settings

BEDROCK_CLIENT = None
OPENAI_CLIENT = None
GOOGLE_CLIENT = None
OLLAMA_CLIENT = None


def bedrock_client():
    global BEDROCK_CLIENT
    if BEDROCK_CLIENT is not None:
        return BEDROCK_CLIENT

    BEDROCK_CLIENT = boto3.client('bedrock-runtime',
                                  aws_access_key_id=os.getenv("AWS_SERVER_PUBLIC_KEY"),
                                  aws_secret_access_key=os.getenv("AWS_SERVER_SECRET_KEY"),
                                  region_name=os.getenv("AWS_REGION")
                                  )
    credentials = BEDROCK_CLIENT._request_signer._credentials

    logging.info(f"Access Key: {credentials.access_key}")
    logging.info(f"Secret Key: {credentials.secret_key[:4] + '...' if credentials.secret_key else 'None'}")
    logging.info(f"Token: {credentials.token[:10] + '...' if credentials.token else 'None'}")
    return BEDROCK_CLIENT

def openai_client():
    global OPENAI_CLIENT
    if OPENAI_CLIENT is not None:
        return OPENAI_CLIENT

    OPENAI_CLIENT = OpenAI(
        organization = os.getenv("OPENAI_ORG_ID"),
        project = os.getenv("OPENAI_PROJECT_ID"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return OPENAI_CLIENT

def google_client():
    global GOOGLE_CLIENT
    if GOOGLE_CLIENT is not None:
        return GOOGLE_CLIENT
    GOOGLE_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return GOOGLE_CLIENT


def ollama_client():
    global OLLAMA_CLIENT
    if OLLAMA_CLIENT is not None:
        return OLLAMA_CLIENT
    OLLAMA_CLIENT = ollama.Client(host=os.getenv("OLLAMA_URL"))
    return OLLAMA_CLIENT



def call_bedrock_titan_llm(model_name: str,
                           prompt: str,
                           max_len: int = None,
                           params: dict = None):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
    # "amazon.titan-text-express-v1"
    if params is None:
        params = {
            "textGenerationConfig": {
                "temperature": 0.3,
                "topP": 1,
                "maxTokenCount": 512
            }
        }
    if max_len is not None:
        params["maxTokenCount"] = max_len
    params["inputText"] = prompt

    response = bedrock_client().invoke_model(
        modelId=model_name,
        body=json.dumps(params)
    )
    response_body = json.loads(response["body"].read())
    completion = response_body['results'][0]["outputText"]
    return completion


def call_bedrock_jurassic2_llm(model_name: str,
                               prompt: str,
                               max_len: int = None,
                               params: dict = None):
    # https://docs.ai21.com/reference/j2-complete-ref
    # "ai21.j2-mid"
    if params is None:
        params = {
            "temperature": 0.3,
            "numResults": 1,
            "maxTokens": 512,
            "topP": 1,
            "topKReturn": 0,    # When using a non-zero value, the response includes the string representations and
            # logprobs for each of the top-K alternatives at each position, in the prompt and in
            # the completions.
        }
    if max_len is not None:
        params["maxTokens"] = max_len
    params["prompt"] = prompt
    response = bedrock_client().invoke_model(
        modelId=model_name,
        body=json.dumps(params)
    )
    response_body = json.loads(response["body"].read())
    completion = response_body["completions"][0]['data']['text']
    return completion


def call_bedrock_jamba_llm(model_name: str,
                           prompt: str,
                           system_prompt: str = None,
                           max_len: int = None,
                           params: dict = None):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jamba.html#model-parameters-jamba-request-response
    if params is None:
        params = {
            "temperature": 0.3,
            "n": 1,             # number of responses
            "max_tokens": 512,
            "top_p": 1,
            "stop" : ["\n\n"]
        }
    if max_len is not None:
        params["max_tokens"] = max_len
    system_massage = {
        "role": "system",
        "content": system_prompt
    }
    user_message = {
        "role": "user",
        "content": prompt
    }
    if system_prompt:
        params["messages"] = [system_massage, user_message]
    else:
        params["messages"] = [user_message]

    response = bedrock_client().invoke_model(
        modelId=model_name,
        body=json.dumps(params)
    )
    response_body = json.loads(response["body"].read())
    completion = response_body["choices"][0]['message']['content']
    return completion


def call_bedrock_llama2_llm(model_name: str,
                            prompt: str,
                            max_len: int = None,
                            params: dict = None):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
    # "meta.llama2-70b-chat-v1"
    if params is None:
        params = {
            "prompt": prompt,
            "temperature": 0.3,
            "top_p": 1,
            "max_gen_len": 512,
        }
    if max_len is not None:
        params["max_gen_len"] = max_len
    params["prompt"] = prompt
    response = bedrock_client().invoke_model(
        modelId=model_name,
        body=json.dumps(params)
    )
    response_body = json.loads(response["body"].read())
    completion = response_body["generation"]
    return completion


def call_bedrock_llama3_llm(model_name: str,
                            prompt: str,
                            max_len: int = None,
                            params: dict = None):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
    return call_bedrock_llama2_llm(model_name=model_name, prompt=prompt, max_len=max_len, params=params)



def call_bedrock_claude_llm(model_name: str,
                            prompt: str,
                            max_len: int = None,
                            params: dict = None):
    # anthropic.claude-instant-v1
    # make sure to use the right prompt format, otherwise there will be an error!
    if params is None:
        params = {
            "max_tokens_to_sample": 1024,
            #"top_k": 10,       # defaults to 250
            "temperature": 0.3,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman:"]
        }
    if max_len is not None:
        params["max_tokens_to_sample"] = max_len
    params["prompt"] = prompt
    response = bedrock_client().invoke_model(
        modelId=model_name,
        body=json.dumps(params)
    )
    response_body = json.loads(response["body"].read())
    completion = response_body["completion"]
    return completion


def call_bedrock_claude3_llm(model_name: str,
                             prompt: str,
                             system_prompt: str = None,
                             max_len: int = None,
                             params: dict = None):
    # anthropic.claude-instant-v1
    # make sure to use the right prompt format, otherwise there will be an error!
    if params is None:
        params = {"anthropic_version": "bedrock-2023-05-31",
                  "max_tokens": 512,
                  "temperature": 0.3,
                  "top_p": 1
                  }
    if max_len is not None:
        params["max_tokens"] = max_len
    if system_prompt is not None:
        params["system"] = system_prompt
    params["messages"] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]

    response = bedrock_client().invoke_model(
        modelId=model_name,
        body=json.dumps(params)
    )
    response_body = json.loads(response["body"].read())
    output_list = response_body.get("content", [])
    try:
        output = output_list[0]["text"]
    except (IndexError, KeyError):
        output = ""
    return output

def call_bedrock_deepseek_llm(model_name: str,
                              prompt: str,
                              max_len: int = None,
                              params: dict = None):
    if params is None:
        params = {
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 1
        }
    if max_len is not None:
        params["max_tokens"] = max_len
    params["prompt"] = prompt

    response = bedrock_client().invoke_model(modelId=model_name, body=json.dumps(params))
    response_body = json.loads(response["body"].read())
    output_list = response_body.get("choices", [])
    try:
        output = output_list[0]["text"]
    except (IndexError, KeyError):
        output = ""
    return output

# def call_sagemaker_llm(endpoint_url: str,
#                        prompt: str,
#                        max_len: int = None,
#                        params: dict = None):
#     if params is None:
#         params = {
#             "max_new_tokens": 512,
#             "return_full_text": False,
#             "do_sample": True,
#             "top_k": 10
#         }
#     if max_len is not None:
#         params["max_new_tokens"] = max_len
#     remote_llm_url = endpoint_url
#     body = {
#         "inputs": prompt,
#         "parameters": params
#     }
#     headers = {'x-api-key': os.getenv('API_KEY')}
#     response = requests.post(
#         url=remote_llm_url, json=body, headers=headers
#     )
#     response.raise_for_status()
#     resp_json = response.json()
#     if 'ErrorCode' in resp_json and resp_json['ErrorCode'] == 'NO_SUCH_ENDPOINT':
#         response.status_code = http.HTTPStatus.FAILED_DEPENDENCY
#         response.reason = "Model Endpoint not reachable"
#         response.raise_for_status()
#     resp_json = response.json()[0]
#     gen_output = resp_json['generated_text']
#     return gen_output

def call_openai_llm(model_name: str,
                    prompt: str,
                    system_prompt: str = None,
                    max_len: int = None,
                    params: dict = None):

    if params is None:
        params = {
            "max_completion_tokens": 512,
            "temperature": 0.3,
            "top_p": 1,
            #  "n": 1,
            # "stop": []
        }
    if max_len is not None:
        params["max_completion_tokens"] = max_len

    system_massage = {
        "role": "system",
        "content": "You are a helpful assistant"
    }
    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }
    if system_prompt:
        params["messages"] = [system_massage, user_message]
    else:
        params["messages"] = [user_message]

    response = openai_client().chat.completions.create(
        model= model_name,
        **params
    )
    output = response.choices[0].message.content
    return output

def call_google_gemini(model_name: str, prompt:str, system_prompt: str = None, max_len: int= None, params: dict = None):
    "gemini-2.0-flash"
    if params is None:
        params = {
            "max_output_tokens": 512,
            "temperature": 0.3,
            #"top_p": 1,
            #  "n": 1,
            # "stop": []
        }
    if max_len is not None:
        params["max_output_tokens"] = max_len

    response = google_client().models.generate_content(
        model= model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            **params
        ),
        contents=[prompt]
    )
    output = response.text
    return output


def call_llm(model_name: str,
             prompt: str,
             instruction_prompt: str = None,
             system_prompt: str = None,
             max_len: int = None) -> str:
    """
    :param model_name:
    :param prompt: the entire prompt including the system message and the user message
    :param max_len: max number of output tokens to be generated
    :param system_prompt:
    :param instruction_prompt:
    :return:
    """
    is_system_message_separate = instruction_prompt is not None and system_prompt is not None
    llm_model = settings.models[model_name]
    platform = llm_model.endpoint_platform.value
    model_name_or_url = llm_model.get_uid()
    #
    logging.info(f"Calling {model_name_or_url} on {platform = }, {max_len = }")
    logging.info(f"Prompt: {prompt}")
    if platform == "bedrock":
        if model_name_or_url.startswith("amazon.titan"):
            gen_out = call_bedrock_titan_llm(model_name=model_name_or_url,
                                             prompt=prompt, max_len=max_len)
        elif model_name_or_url.startswith("ai21.j2"):
            gen_out = call_bedrock_jurassic2_llm(model_name=model_name_or_url,
                                                 prompt=prompt, max_len=max_len)
        elif model_name_or_url.startswith("ai21.jamba"):
            gen_out = call_bedrock_jamba_llm(model_name=model_name_or_url,
                                             prompt=prompt, max_len=max_len)
        elif model_name_or_url.startswith("meta.llama2"):
            gen_out = call_bedrock_llama2_llm(model_name=model_name_or_url,
                                              prompt=prompt, max_len=max_len)
        elif model_name_or_url.startswith("meta.llama3") or model_name_or_url[3:].startswith("meta.llama3"):
            gen_out = call_bedrock_llama3_llm(model_name=model_name_or_url,
                                              prompt=prompt, max_len=max_len)
        elif model_name_or_url.startswith("anthropic.claude-3") or model_name_or_url[3:].startswith("anthropic.claude-3"):
            gen_out = call_bedrock_claude3_llm(model_name=model_name_or_url,
                                               prompt=instruction_prompt if is_system_message_separate else prompt ,
                                               system_prompt=system_prompt, max_len=max_len)
        elif model_name_or_url.startswith("anthropic.claude"):
            gen_out = call_bedrock_claude_llm(model_name=model_name_or_url,
                                              prompt=prompt, max_len=max_len)
        elif model_name_or_url.startswith("us.deepseek"): # does not exist in eu yet
            gen_out = call_bedrock_deepseek_llm(model_name=model_name_or_url,
                                                prompt=prompt, max_len=max_len)
        else:
            raise ValueError(f"{model_name_or_url} not implemented for platform {platform}")
    # elif platform == "sagemaker":
    #     gen_out = rl.call_sagemaker_llm(endpoint_url=model_name_or_url,
    #                                     prompt=prompt, max_len=max_len)
    elif platform == "openai":
        gen_out = call_openai_llm(model_name=model_name_or_url,
                                  prompt=instruction_prompt if is_system_message_separate else prompt,
                                  system_prompt=system_prompt,
                                  max_len=max_len)
    elif platform == "google":
        gen_out = call_google_gemini(model_name=model_name_or_url,
                                     prompt=instruction_prompt if is_system_message_separate else prompt,
                                     system_prompt=system_prompt,
                                     max_len=max_len)
    elif platform == "ollama":
        response = ollama_client().generate(model_name_or_url,
                                            prompt=instruction_prompt if is_system_message_separate else prompt,
                                            system=system_prompt)
        gen_out = response.response
    else:
        raise ValueError(f"Platform {platform} unknown")
    logging.info(f"Generated output: {gen_out}")
    return gen_out
