import logging
import os

import boto3
import ollama
from google import genai
from openai import OpenAI

BEDROCK_CLIENT = None
OPENAI_CLIENT = None
GOOGLE_CLIENT = None
OLLAMA_CLIENT = None


def bedrock_client():
    global BEDROCK_CLIENT
    if BEDROCK_CLIENT is not None:
        return BEDROCK_CLIENT

    if not os.getenv("AWS_SERVER_SECRET_KEY"):
        raise EnvironmentError("Please set AWS_SERVER_SECRET_KEY environment variable before using AWS.")
    else:
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

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Please set OPENAI_API_KEY environment variable before using OPENAI.")
    else:
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

    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("Please set GEMINI_API_KEY environment variable before using GCP.")
    else:
        GOOGLE_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return GOOGLE_CLIENT


def ollama_client():
    global OLLAMA_CLIENT
    if OLLAMA_CLIENT is not None:
        return OLLAMA_CLIENT

    if not os.getenv("OLLAMA_URL"):
        raise EnvironmentError("Please set OLLAMA_URL environment variable before using ollama.")
    else:
        OLLAMA_CLIENT = ollama.Client(host=os.getenv("OLLAMA_URL"))
        return OLLAMA_CLIENT
