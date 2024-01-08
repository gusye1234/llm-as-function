import asyncio
from llm_as_function import gpt35_func
import json
from pydantic import BaseModel, Field


class Result(BaseModel):
    weather_summary: str = Field(description="The summary of the weather")


class GetCurrentWeatherRequest(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")


def get_current_weather(request: GetCurrentWeatherRequest):
    """
    Get the current weather in a given location
    """
    weather_info = {
        "location": request.location,
        "temperature": "72",
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


@gpt35_func.func(get_current_weather)
def fool() -> Result:
    """
    你需要调用get_current_weather函数, 然后汇报下New York. 请注意, 你需要对天气情况进行文字总结
    注意, 一个函数最多调用一次
    """
    pass


@gpt35_func.func(get_current_weather).async_call
def fool2() -> Result:
    """
    你需要调用get_current_weather函数, 然后汇报下New York. 请注意, 你需要对天气情况进行文字总结
    注意, 一个函数最多调用一次
    """
    pass


def test_fn_calling():
    result = fool().unpack()
    print(result)


def test_fn_async_calling():
    result = asyncio.run(fool2()).unpack()
    print(result)
