import asyncio
from pydantic import BaseModel, Field
from llm_as_function import gpt35_func


class Result(BaseModel):
    emoji: str = Field(description="随机输出一个emoji")


@gpt35_func
def fool() -> Result:
    """
    你需要随机输出一个emoji
    """
    pass


@gpt35_func.async_call
def fool2() -> Result:
    """
    你需要随机输出一个emoji
    """
    pass


def test_sync():
    result = fool().unpack()
    assert result["emoji"] != ""


def test_async():
    result = asyncio.run(fool2()).unpack()
    assert result["emoji"] != ""
