import os
import sys

sys.path.append("../")
os.environ["LEVEL"] = "DEBUG"
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


# ! Only support openai model yet, ernie(文心一言) is not supported.
@gpt35_func.func(get_current_weather)
def fool() -> Result:
    """
    Search the weather of New York. And then summarize the weather situation.
    Be careful, you should not call a function twice.
    """
    pass


def fn_calling():
    result = fool().unpack()
    print(result)


if __name__ == "__main__":
    fn_calling()
