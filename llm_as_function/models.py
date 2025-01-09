from typing import Literal
import httpx
import ollama

import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from functools import wraps

from llm_as_function.llm_func import RuntimeOptions, empty_runtime_options
from .utils import logger

# Prompts for different providers
ERNIE_PROMPT = """

你必须要按照如下的JSON格式代码输出你的结果, 否则你的结果将无法被评测系统正确解析:
{json_schema}
"""

DEFAULT_PROMPT = """

!!! You must output your results in the following JSON schema. Strictly adhere to the following JSON schema, and do not return a list in this format:
{json_schema}
"""

OLLAMA_LLAMA2_PROMPT = """

You are the JSON Machine return the asked query in a json format like below. MAKE SURE TO FOLLOW THE TYPES AND STRUCTURE STRICTLY, (ADD QUOTES TO STRING) OTHERWISE YOUR RESULT WILL NOT BE EVALUATED CORRECTLY:
{json_schema}
"""


def get_json_schema_prompt(provider: Literal["ernie", "openai", "ollama"] | str, model: str | None = None) -> str:
    match [provider, model]:
        case ["ernie", None]:
            return ERNIE_PROMPT
        case ["ollama", "llama2"]:
            return OLLAMA_LLAMA2_PROMPT  # The llama2 is a smaller model, so it needs more guidance on it's prompt
        case _:
            return DEFAULT_PROMPT


def openai_max_retry(func, retry_times=3):
    @wraps(func)
    def new_func(*args, **kwargs):
        current_retries = 0
        while current_retries < retry_times:
            try:
                result = func(*args, **kwargs)
                return result
            except openai.APIConnectionError as e:
                current_retries += 1
                if current_retries >= retry_times:
                    raise e
                logger.warning(
                    f"Connect error for {func.__name__}, retry {current_retries} times"
                )
            except Exception as e:
                raise e

    return new_func


def async_openai_max_retry(func, retry_times=3):
    @wraps(func)
    async def new_func(*args, **kwargs):
        current_retries = 0
        while current_retries < retry_times:
            try:
                result = await func(*args, **kwargs)
                return result
            except openai.APIConnectionError as e:
                current_retries += 1
                if current_retries >= retry_times:
                    raise e
                logger.warning(
                    f"Connect error for {func.__name__}, retry {current_retries} times"
                )
            except Exception as e:
                raise e

    return new_func


def ollama_max_retry(func, retry_times=3):
    @wraps(func)
    def new_func(*args, **kwargs):
        current_retries = 0
        while current_retries < retry_times:
            try:
                result = func(*args, **kwargs)
                return result
            except httpx.ConnectError as e:
                current_retries += 1
                if current_retries >= retry_times:
                    raise e
                logger.warning(
                    f"Connect error for {func.__name__}, retry {current_retries} times"
                )
            except Exception as e:
                raise e

    return new_func


def async_ollama_max_retry(func, retry_times=3):
    @wraps(func)
    async def new_func(*args, **kwargs):
        current_retries = 0
        while current_retries < retry_times:
            try:
                result = await func(*args, **kwargs)
                return result
            except httpx.ConnectError as e:
                current_retries += 1
                if current_retries >= retry_times:
                    raise e
                logger.warning(
                    f"Connect error for {func.__name__}, retry {current_retries} times"
                )
            except Exception as e:
                raise e

    return new_func


@openai_max_retry
def openai_single_create(
    query,
    client: OpenAI,
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    function_messages=[],
    runtime_options: RuntimeOptions = empty_runtime_options(),
) -> ChatCompletion:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}] + function_messages,
        temperature=temperature,
        response_format={"type": "json_object"},
        # This is the same type as list[ChatCompletionToolParams] but since we user our own types instead of openai's, we need to ignore this
        tools=runtime_options["tools"],  # type: ignore
        tool_choice=runtime_options["tool_choice"],
    )
    return response


@async_openai_max_retry
async def openai_single_acreate(
    query,
    client: AsyncOpenAI,
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    function_messages=[],
    runtime_options: RuntimeOptions = empty_runtime_options(),
) -> ChatCompletion:

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}] + function_messages,
        temperature=temperature,
        response_format={"type": "json_object"},
        # This is the same type as list[ChatCompletionToolParams] but since we user our own types instead of openai's, we need to ignore this
        tools=runtime_options["tools"],  # type: ignore
        tool_choice=runtime_options["tool_choice"],
    )
    return response


@ollama_max_retry
def ollama_single_create(
    query,
    client: ollama.Client,
    model="llama2",
    temperature=0.1,
    function_messages=[],
    runtime_options: RuntimeOptions = empty_runtime_options(),
) -> ollama.ChatResponse:

    if runtime_options["tools"]:
        repsonse = client.chat(
            model=model,
            messages=[{"role": "user", "content": query}] + function_messages,
            options={"temperature": temperature},
            tools=runtime_options["tools"],  # ollama can just take in python functions but it also supports the tool format
            format=runtime_options["output_schema"],
        )
    else:
        repsonse = client.chat(
            model=model,
            messages=[{"role": "user", "content": query}] + function_messages,
            options={"temperature": temperature},
            format=runtime_options["output_schema"],
        )

    return repsonse


@async_ollama_max_retry
async def ollama_single_acreate(
    query,
    client: ollama.AsyncClient,
    model="llama2",
    temperature=0.1,
    function_messages=[],
    runtime_options: RuntimeOptions = empty_runtime_options(),
) -> ollama.ChatResponse:

    if runtime_options["tools"]:
        response = await client.chat(
            model=model,
            messages=[{"role": "user", "content": query}] + function_messages,
            options={"temperature": temperature},
            tools=runtime_options["tools"],  # ollama can just take in python functions but it also supports the tool format
            format=runtime_options["output_schema"],
        )
    else:
        response = await client.chat(
            model=model,
            messages=[{"role": "user", "content": query}] + function_messages,
            options={"temperature": temperature},
            format=runtime_options["output_schema"],
        )

    return response
