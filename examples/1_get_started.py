import os
import sys
from dotenv import load_dotenv

sys.path.append("../")
load_dotenv()

import erniebot
from pydantic import BaseModel, Field
from llm_as_function import gpt35_func

erniebot.api_type = "aistudio"
erniebot.access_token = os.environ["ERNIE_KEY"]


class Result(BaseModel):
    emoji: str = Field(description="随机输出一个emoji")


@gpt35_func
def fool() -> Result:
    """
    你需要随机输出一个emoji
    """
    pass


@gpt35_func
def fool2(emotion) -> Result:
    """
    你需要随机输出一个emoji, 我希望感情是{emotion}
    """
    pass


# print(fool())
print(fool2(emotion="开心"))
