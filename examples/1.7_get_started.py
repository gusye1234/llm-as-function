import sys
from dotenv import load_dotenv

sys.path.append("../")
load_dotenv()

from pydantic import BaseModel, Field
from llm_as_function import llama2_func


class Result(BaseModel):
    emoji: str = Field(description="The emoji you output")


@llama2_func
def fool() -> Result:  # type: ignore
    """You need to randomly output an emoji"""
    pass


@llama2_func
def fool2(emotion) -> Result:  # type: ignore
    """
    You need to randomly output an emoji, which should be {emotion}
    """
    pass


print(fool2(emotion="Happy Smiling Yellow"))
