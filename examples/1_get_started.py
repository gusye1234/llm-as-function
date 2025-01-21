import os
import sys
import asyncio
from dotenv import load_dotenv

sys.path.append("../")
load_dotenv()

from pydantic import BaseModel, Field
from llm_as_function import gpt35_func


class Result(BaseModel):
    emoji: str = Field(description="The emoji you output")


@gpt35_func
def fool() -> Result:  # type: ignore
    """You need to randomly output an emoji"""
    pass


@gpt35_func
def fool2(emotion: str) -> Result:  # type: ignore
    """
    You need to randomly output an emoji, which should be {emotion}
    """
    pass


print(fool2(emotion="Happy"))
