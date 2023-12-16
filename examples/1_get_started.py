import os
import sys
from dotenv import load_dotenv

sys.path.append("../")
load_dotenv()

import erniebot
from pydantic import BaseModel, Field
from llm_as_function import LLMFunc

erniebot.api_type = "aistudio"
erniebot.access_token = os.environ["ERNIE_KEY"]


class Result(BaseModel):
    emoji: str = Field(description="随机输出一个emoji")


@LLMFunc()
def fool() -> Result:
    """
    你需要随机输出一个emoji
    """
    pass


print(fool())
