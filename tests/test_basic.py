import asyncio
from pydantic import BaseModel, Field
from llm_as_function import gpt35_func


class Result(BaseModel):
    emoji: str = Field(description="随机输出一个emoji")


def test_sync():
    @gpt35_func
    def fool() -> Result:  # type: ignore
        """
        你需要随机输出一个emoji
        """
        pass

    result = fool().unpack()
    assert result["emoji"] != ""  # type: ignore


def test_async():
    @gpt35_func.async_call
    def fool2() -> Result:  # type: ignore
        """
        你需要随机输出一个emoji
        """
        pass

    result = asyncio.run(fool2()).unpack()  # type: ignore
    assert result["emoji"] != ""
