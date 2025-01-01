import os
import sys
from dotenv import load_dotenv
from rich import print

sys.path.append("../")
load_dotenv()

from pydantic import BaseModel, Field
from llm_as_function import llama3_1_func, Final


from time import sleep


class Result(BaseModel):
    value: int = Field(description="斐波那契数列计算的值")


@llama3_1_func
def f(x: int) -> Result:
    """You need to calculate the {x}th term of the Fibonacci sequence.
    Given that you have the values of the two preceding terms, which are {a} and {b}. You calculate the {x}th term by adding the values of the two preceding terms. Please compute the value of the {x}th term.
    """
    if x == 1 or x == 0:
        return Final({"value": x})  # type: ignore

    print(f"Running {x-1}")
    a = f(x=x - 1)

    print(f"Running {x-2}")
    b = f(x=x - 2)

    # if a.ok it means the unpack is a dict so ignore the type error
    a_value = a.unpack()["value"] if a.ok() else a.unpack()  # type: ignore
    b_value = b.unpack()["value"] if b.ok() else b.unpack()  # type: ignore

    return {"a": a_value, "b": b_value}  # type: ignore


print(f(x=3))
