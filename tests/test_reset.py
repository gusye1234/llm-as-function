import json
from llm_as_function import gpt35_func
from copy import copy, deepcopy
from pydantic import BaseModel, Field


class Fool(BaseModel):
    a: str = Field(description="a")


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


def test_reset():
    original_func = copy(gpt35_func)

    @gpt35_func
    def fool() -> Fool:  # type: ignore
        """
        You need to randomly output an emoji
        """

    assert original_func == gpt35_func


def test_func_reset():
    @gpt35_func.func(get_current_weather)
    def fool() -> Fool:  # type: ignore
        """
        You need to randomly output an emoji
        """

    assert gpt35_func.runtime_options == {}
