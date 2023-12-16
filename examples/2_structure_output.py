import os
import sys
from dotenv import load_dotenv

sys.path.append("../")
load_dotenv()

import erniebot
from rich import print
from pydantic import BaseModel, Field
from llm_as_function import ernie_func

erniebot.api_type = "aistudio"
erniebot.access_token = os.environ["ERNIE_KEY"]


class Reason(BaseModel):
    where: str = Field(description="这个emoji可以用在哪些地方?")
    warning: str = Field(description="我使用这个emoji需要注意什么吗")


class StructuredOutput(BaseModel):
    emoji: str = Field(description="随机输出的emoji")
    why: str = Field(description="为什么输出这个emoji")
    more: Reason = Field(description="更多关于这个emoji的信息")


class Result(BaseModel):
    emoji: StructuredOutput = Field(description="随机输出一个emoji和他的相关的信息")


@ernie_func
def fool() -> Result:
    """
    你需要随机输出一个emoji
    """
    pass


print(fool())
