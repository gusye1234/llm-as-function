from llm_as_function import gpt35_func
from copy import copy
from pydantic import BaseModel, Field


class Fool(BaseModel):
    a: str = Field(description="a")


def test_reset():
    original_func = copy(gpt35_func)

    @gpt35_func
    def fool() -> Fool:
        """
        You need to randomly output an emoji
        """

    assert original_func == gpt35_func
