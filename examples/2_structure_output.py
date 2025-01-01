import os
import sys
from dotenv import load_dotenv

sys.path.append("../")
load_dotenv()

from rich import print
from pydantic import BaseModel, Field
from llm_as_function import gpt35_func


class Reason(BaseModel):
    where: str = Field(description="Where I can use this emoji?")
    warning: str = Field(description="Anything I should notice to use this emoji?")


class StructuredOutput(BaseModel):
    emoji: str = Field(description="The emoji")
    why: str = Field(description="Why you choose this emoji?")
    more: Reason = Field(description="More information about this emoji")


class Result(BaseModel):
    emoji: StructuredOutput = Field(description="The emoji and its related information")


@gpt35_func
def fool() -> Result:  # type: ignore
    """
    You need to randomly output an emoji
    """
    pass


print(fool())
