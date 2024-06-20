import os
import sys

sys.path.append("../")
os.environ["LEVEL"] = "DEBUG"
from llm_as_function import gpt35_func
import json
import asyncio
from pydantic import BaseModel, Field


class Result(BaseModel):
    summary: str = Field(description="The response summary sentence")


class GetCurrentWeatherRequest(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")


class GetCurrentTimeRequest(BaseModel):
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


def get_current_time(request: GetCurrentTimeRequest):
    """
    Get the current time in a given location
    """
    time_info = {
        "location": request.location,
        "time": "2024/1/1",
    }
    return json.dumps(time_info)


@gpt35_func.func(get_current_weather).func(get_current_time)
def fool() -> Result:
    """
    Search the weather and current time of New York. And then summary the time and weather in one sentence.
    Be careful, you should not call the same function twice.
    """
    pass


@gpt35_func.func(get_current_weather).func(get_current_time).async_call
def fool2() -> Result:
    """
    Search the weather and current time of New York. And then summary the time and weather in one sentence.
    Be careful, you should not call the same function twice.
    """
    pass


def fn_calling():
    result = fool().unpack()
    print(result)
    result = asyncio.run(fool2()).unpack()
    print(result)


if __name__ == "__main__":
    fn_calling()
