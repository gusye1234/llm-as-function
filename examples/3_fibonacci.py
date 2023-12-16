import os
import sys
from dotenv import load_dotenv

sys.path.append("../")
load_dotenv()

import erniebot
from pydantic import BaseModel, Field
from llm_as_function import LLMFunc, Final

erniebot.api_type = "aistudio"
erniebot.access_token = os.environ["ERNIE_KEY"]


from time import sleep


class Result(BaseModel):
    value: int = Field(description="斐波那契数列计算的值")


@LLMFunc()
def f(x: int) -> Result:
    """
    你需要计算斐波那契数列的第{x}项, 你有他的前两项的值, 分别是{a}和{b}. 你计算第{x}项的方式是将前两项的值相加. 请你计算出第{x}项的值"""
    if x == 1 or x == 0:
        return Final({"value": x})
    a = f(x=x - 1)
    sleep(1)
    b = f(x=x - 2)
    sleep(1)
    return {"a": a["value"], "b": b["value"]}


print(f(x=3))
