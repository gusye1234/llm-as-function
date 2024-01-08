import openai
from openai import OpenAI, AsyncOpenAI
import erniebot
from .utils import logger

JSON_SCHEMA_PROMPT = {
    "ernie": """

你必须要按照如下的JSON格式代码输出你的结果, 否则你的结果将无法被评测系统正确解析:
{json_schema}
""",
    "openai": """

!!! You must output your results in the following JSON schema:
{json_schema}
""",
}


def ernie_single_create(
    query, model="ernie-bot-4", max_retry=3, temperature=0.1, runtime_options={}
):
    retry = 0
    while retry < max_retry:
        try:
            response = erniebot.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                temperature=temperature,
                **runtime_options,
            )
            return response.get_result()
        except erniebot.errors.APIError:
            retry += 1
            logger.warning(f"Failed {retry} times, retrying...")
    raise erniebot.errors.APIError("Max retry is reached")


async def ernie_single_acreate(
    query, model="ernie-bot-4", max_retry=3, temperature=0.1, runtime_options={}
):
    retry = 0
    while retry < max_retry:
        try:
            response = await erniebot.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": query}],
                temperature=temperature,
                **runtime_options,
            )
            return response.get_result()
        except erniebot.errors.APIError:
            retry += 1
            logger.warning(f"Failed {retry} times, retrying...")
    raise erniebot.errors.APIError("Max retry is reached")


def openai_single_create(
    query,
    client: OpenAI,
    model="gpt-3.5-turbo-1106",
    max_retry=3,
    temperature=0.1,
    function_messages=[],
    runtime_options={},
):
    retry = 0
    while retry < max_retry:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}] + function_messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                **runtime_options,
            )
            return response
        except Exception as e:
            logger.error(e)
            logger.warning(f"Failed {retry} times, retrying...")
    raise openai.APIConnectionError("Max retry is reached")


async def openai_single_acreate(
    query,
    client: AsyncOpenAI,
    model="gpt-3.5-turbo-1106",
    max_retry=3,
    temperature=0.1,
    function_messages=[],
    runtime_options={},
):
    retry = 0
    while retry < max_retry:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}] + function_messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                **runtime_options,
            )
            return response
        except Exception as e:
            logger.error(e)
            logger.warning(f"Failed {retry} times, retrying...")
            retry += 1
    raise openai.APIConnectionError("Max retry is reached")
