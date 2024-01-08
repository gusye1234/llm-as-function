import os
import sys
import asyncio
from dotenv import load_dotenv

sys.path.append("../")
load_dotenv()

from pydantic import BaseModel, Field
from llm_as_function import gpt35_func, LLMFunc


class Result(BaseModel):
    emoji: str = Field(description="The emoji you output")


@gpt35_func.async_call
def fool2(emotion) -> Result:
    """
    You need to output an emoji, which is {emotion}
    """
    pass


async def async_call2():
    result = await asyncio.gather(
        *[
            asyncio.create_task(fool2(emotion="happy")),
            asyncio.create_task(fool2(emotion="sad")),
            asyncio.create_task(fool2(emotion="weird")),
        ]
    )
    print([r.unpack() for r in result])


# asyncio.run(async_call2())

set_bound_func = LLMFunc(async_max_time=2)

print(set_bound_func.async_funcs)


@set_bound_func.async_call
def fool3(emotion) -> Result:
    """
    You need to output an emoji, which is {emotion}
    """
    pass


async def async_call3():
    result = await asyncio.gather(
        *[
            asyncio.create_task(fool3(emotion="happy")),
            asyncio.create_task(fool3(emotion="sad")),
            asyncio.create_task(fool3(emotion="weird")),
        ]
    )
    print([r.unpack() for r in result])


asyncio.run(async_call3())
