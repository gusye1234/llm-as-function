<div align="center">
  <h1>llm-as-function</h1>
  <p><strong>Embed LLM into your python function</strong></p>
  <p>
        <a href="https://pypi.org/project/llm-as-function/">
      <img src="https://img.shields.io/pypi/v/llm-as-function.svg">
    </a>
  </p>
</div>


--------------------------------------------------------------------------------

[English](./README.md) | [简体中文](./README_cn.md)



`llm-as-function` is a Python library that helps you quickly build functions based on Large Language Models (LLMs). You can use `LLMFunc` as a decorator for your functions, while also providing type annotations and writing docstrings for your functions. `llm-as-function` will automatically complete the parameter filling by invoking the large model and return formatted output.

> `llm-as-function` 是一个帮助你快速构建基于LLM的函数的Python库. 你可以使用`LLMFunc`作为你函数的装饰器, 同时给你的函数进行类型标注和docstring的编写, `llm-as-function`会自动的通过调用大模型完成参数的填入, 并且返回格式化的输出. 

The `llm-as-function` also supports defining code bodies within the LLM function for precise inference control and business logic.

> `llm-as-function`还支持在LLM函数中定义代码体, 用于精确的推理控制和业务逻辑.

## Get Started
```
pip install llm-as-function
```
## Features

### Basic usage

```python
from llm_as_function import gpt35_func # gpt35_func using gpt-3.5-turbo as the LLM backend
from pydantic import BaseModel, Field

# Validate and Define output types with Pydantic
class Result(BaseModel):
    emoji: str = Field(description="The output emoji")

# Using decorators, LLMFunc will automatically recognize the input and output of your function, as well as the function's docstring.
# Here, the function's DocString is your Prompt, so please design it carefully.
@gpt35_func
def fool() -> Result:
    """
    You need to randomly output an emoji
    """
    pass
  
print(foo()) # {emoji: "😅"}
```

You can construct more complex output logic.

```python
class Reason(BaseModel):
    where: str = Field(description="Where I can use this emoji?")
    warning: str = Field(description="Anything I should notice to use this emoji?")

class StructuredOutput(BaseModel):
    emoji: str = Field(description="The emoji")
    why: str = Field(description="Why you choose this emoji?")
    more: Reason = Field(description="More information about this emoji")

class Result(BaseModel):
    emoji: StructuredOutput = Field(description="The emoji and its related information")
    
@gpt35_func
def fool() -> Result:
    """
    You need to randomly output an emoji
    """
    pass

print(fool())
```

You will get the below output:

```python
Final(
    pack={
        'emoji': {
            'emoji': '🎉',
            'why': 'I choose this emoji because it represents celebration and excitement.',
            'more': {
                'where': 'I can use this emoji to express joy and happiness in messages or social media
posts.',
                'warning': 'Be mindful of the context in which you use this emoji, as it may not be appropriate
for all situations.'
            }
        }
    },
    raw_response=None
)

```

### Prompt variables

You can also dynamically insert variables into the prompt.

```python
@gpt35_func
def fool2(emotion) -> Result:
    """
    You need to randomly output an emoji, the emoji should be {emotion}
    """
    pass
  
print(foo2(emotion="Happy")) # {'emoji': '😊'}
```

### Merge program with LLM

**The most crucial part** is that you can insert `python` code into your function, which will run before the actual LLM execution, so you can accomplish similar tasks:

```python
@gpt35_func
def fool() -> Result:
    """
    You need to output an emoji
    """
    print("Logging once")
```

More interestingly, you can invoke other functions within it, other LLM functions, such as calling itself (refer to `examples/3_fibonacci.py`):

```python
from llm_as_function import gpt35_func, Final
class Result(BaseModel):
    value: int = Field(description="The calculated value of the Fibonacci sequence.")

@gpt35_func
def f(x: int) -> Result:
    """You need to calculate the {x}th term of the Fibonacci sequence, given that you have the values of the two preceding terms, which are {a} and {b}. The method to calculate the {x}th term is by adding the values of the two preceding terms. Please compute the value of the {x}th term."""
    if x == 1 or x == 0:
        # The `Final` is a class in `llm-as-function`, and returning this class indicates that you do not need the large model to process your output. Inside `Final` should be a dictionary, and its format should be the same as the `Result` you defined.
        return Final({"value": x})
    a = f(x=x - 1)
    b = f(x=x - 2)
    # A normal function return indicates that you have passed 'local variables' to the large model, and the variables you return will be inserted into your prompt.
    return {"a": a.unpack()["value"], "b": b.unpack()["value"]}

print(f(3)) # {value: 2}
```

### Function calling

`llm-as-function` offer the similar way to tell LLM a series of function tools(`examples/5_function_calling.py`)

```python
class Result(BaseModel):
    summary: str = Field(description="The response summary sentence")


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
    Search the weather of New York. And then summary the weather in one sentence.
    Be careful, you should not call the same function twice.
    """
    pass
```

[Parallel function calling](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling) for OpenAI is supported:

```python
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
```

### Async Call

Async calling for LLM api is supported, you call simply add `async_call` then the function will be an async python function(`examples/1.5_get_started.py`):

```python
@gpt35_func.async_call
def fool(emotion) -> Result:
    """
    You need to output an emoji, which is {emotion}
    """
    pass
    
async def async_call():
    result = await asyncio.gather(
        *[
            asyncio.create_task(fool(emotion="happy")),
            asyncio.create_task(fool(emotion="sad")),
            asyncio.create_task(fool(emotion="weird")),
        ]
    )
    print([r.unpack() for r in result])
```

### Ollama Models Support

`llm-as-function` supports various Ollama models with structured output capabilities:

```python
from llm_as_function import llama2_func
from pydantic import BaseModel, Field

class EmojiCharacter(BaseModel):
    emoji: str = Field(description="The emoji character")
    name: str = Field(description="Name of the emoji character")
    personality: str = Field(description="Personality traits of the emoji")

class EmojiStory(BaseModel):
    character1: EmojiCharacter
    character2: EmojiCharacter
    plot: str = Field(description="A short story about the interaction between the two emojis")
    moral: str = Field(description="The moral of the story")

@llama2_func
def generate_emoji_story(theme: str) -> EmojiStory:
    """
    Create a short story about two emoji characters based on the given theme.
    The story should include their personalities and a moral lesson.
    """
    pass

# Usage
story = generate_emoji_story(theme="friendship")
print(story.unpack())
```

All Ollama models (`llama2_func`, `llama3_func`, etc.) support structured output, and models like `llama3_1_func`, `llama3_3_func`, and `llama3_2_1b_func` also support tool calling functionality.

More demos in `examples/`

## Docs

`LLMFunc`

```python
# LLMFunc supports both OpenAI and Ollama providers

# OpenAI configuration
@LLMFunc(model="gpt-3.5-turbo-1106", temperature=0.3, openai_base_url=..., openai_api_key=...)
def fool() -> Result:
    ...

# Ollama configuration
@LLMFunc(model="llama2", temperature=0.3, has_structured_output=True)
def fool() -> Result:
    ...

-----------------------------------------------------
# For your convenience, llm-as-function already instantiated common LLMFunc providers
from llm_as_function import (
    # OpenAI models
    gpt35_func, gpt4_func,
    # Ollama models
    llama2_func, llama3_func, llama3_1_func,
    qwen2_1_5b_func
)

@gpt35_func  # or @llama2_func
def fool() -> Result:
    pass
-----------------------------------------------------
# Parse mode: ["error", "accept_raw"], default "error"
# llm-as-function may fail to return the result format, due to LLM doesn't always obey
# When the parsing fails, there are two mode you can choose:

@LLMFunc(parse_mode="error")
def fool() -> Result:
    ...
result = fool() # When the parsing fails, fool will raise an error

@LLMFunc(parse_mode="accept_raw")
def fool() -> Result:
    ...
result = fool() # When the parsing fails, fool will not raise an error but return the raw response of LLM, refer to the `Final` class
```

`Final`

```python
# The return value of any llm function is a `Final` class

result: Final = fool()
if result.ok():
  format_response = result.unpack() # the response will be a formated dict
else:
  raw_response = result.unpack() # the response will be the raw string result from LLM
```

## FQA

* The formatting of the return from `llm-as-function` depends on the capabilities of the model you are using. Sometimes, larger models may not be able to return a parsable JSON format, which can lead to an Error or return the raw response if you set the `parse_mode="accept_raw"`.


  代表你遇到rate limit限制了, 考虑进行换模型或者在每次使用function后sleep一段时间
