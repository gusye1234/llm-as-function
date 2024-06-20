import openai
from openai import OpenAI, AsyncOpenAI
from functools import wraps
from .utils import logger

JSON_SCHEMA_PROMPT = {
    "ernie": """

你必须要按照如下的JSON格式代码输出你的结果, 否则你的结果将无法被评测系统正确解析:
{json_schema}
""",
    "openai": """

!!! You must output your results in the following JSON schema. Strictly adhere to the following JSON schema, and do not return a list in this format:
{json_schema}
""",
}


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


@openai_max_retry
def openai_single_create(
    query,
    client: OpenAI,
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    function_messages=[],
    runtime_options={},
):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}] + function_messages,
        temperature=temperature,
        response_format={"type": "json_object"},
        **runtime_options,
    )
    return response


@async_openai_max_retry
async def openai_single_acreate(
    query,
    client: AsyncOpenAI,
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    function_messages=[],
    runtime_options={},
):

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}] + function_messages,
        temperature=temperature,
        response_format={"type": "json_object"},
        **runtime_options,
    )
    return response
