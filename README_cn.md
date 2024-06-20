<div align="center">
  <h1>llm-as-function</h1>
  <p><strong>LLM即函数：基于嵌入式LLM的Python编程框架</strong></p>
  <p>
        <a href="https://pypi.org/project/llm-as-function/">
      <img src="https://img.shields.io/pypi/v/llm-as-function.svg">
    </a>
  </p>
</div>


--------------------------------------------------------------------------------

[English](./README.md) | [简体中文](./README_cn.md)

`llm-as-function` 是一个帮助你快速构建基于LLM的函数的Python库. 你可以使用`LLMFunc`作为你函数的装饰器, 同时给你的函数进行类型标注和docstring的编写, `llm-as-function`会自动的通过调用大模型完成参数的填入, 并且返回格式化的输出. 

`llm-as-function`还支持在LLM函数中定义代码体, 用于精确的推理控制和业务逻辑.

## 快速开始
```
pip install llm-as-function
```
## 使用方式

### 基础用法

```python
from llm_as_function import gpt35_func # 使用gpt-3.5-turbo作为LLM推理后端
from pydantic import BaseModel, Field

# 定义输出格式
class Result(BaseModel):
    emoji: str = Field(description="The output emoji")

# 使用装饰器装饰你的函数，LLM会自动识别你的输入输出信息，并将你的注释信息看作Prompt
@gpt35_func
def fool() -> Result:
    """
    You need to randomly output an emoji
    """
    pass
  
print(foo()) # {emoji: "😅"}
```

你同样可以组织一些更为复杂的输出结构：

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
"""
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
"""
```

### 动态Prompt变量

你同样可以在prompt中动态嵌入一些可变信息

```python
@gpt35_func
def fool2(emotion) -> Result:
    """
    You need to randomly output an emoji, the emoji should be {emotion}
    """
    pass
  
print(foo2(emotion="Happy")) # {'emoji': '😊'}
```

### 代码与LLM有机结合

**更关键的是**你可以在函数中写入python语句，这些语句会在调用LLM之前执行，例如，下列代码在执行LLM之前输出一段日志信息。

```python
@gpt35_func
def fool() -> Result:
    """
    You need to output an emoji
    """
    print("Logging once")
```

另外，你还可以在一个被LLM封装的函数中调用其他被LLM封装的函数，例如递归调用。（参考`examples/3_fibonacci.py`）

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

`llm-as-function` 提供了一种相似的嵌入大模型工具函数的方式(`examples/5_function_calling.py`)

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

OpenAI的[Parallel function calling](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling)也支持:

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

Async calling也是支持的, 你只需要简单的在装饰语句中添加 `async_call` , llm function便是一个async python函数 (`examples/1.5_get_started.py`):

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

更多样例请参考`examples/`

## 详细介绍

### `LLMFunc`

```python
# LLMFunc(LLM封装器)当前仅支持OpenAI provider
@LLMFunc(model="gpt-3.5-turbo-1106", temperature=0.3, openai_base_url=..., openai_api_key=...)
def fool() -> Result:
    ...

-----------------------------------------------------
# 方便起见，本项目已经实例化一些封装器，你可以直接使用gpt35_func, gpt4_func
from llm_as_function import gpt35_func, gpt4_func

@gpt35_func
def fool() -> Result:
    pass
-----------------------------------------------------
# 解析模式共两种: ["error", "accept_raw"], 默认为 "error"
# llm-as-function可能无法永远遵循输出格式要求（这取决于对应的LLM的性能）
# 当解析出错时，这两种模式会产生不同的结果：

@LLMFunc(parse_mode="error")
def fool() -> Result:
    ...
result = fool() # 解析出错时，fool会抛出异常

@LLMFunc(parse_mode="accept_raw")
def fool() -> Result:
    ...
result = fool() # 解析出错时，fool不会抛出异常，但是会返回LLM的原始回复，更多细节见`Final`模块
```

### `Final`

```python
# The return value of any llm function is a `Final` class

result: Final = fool()
if result.ok():
  format_response = result.unpack() # the response will be a formated dict
else:
  raw_response = result.unpack() # the response will be the raw string result from LLM
```

## FQA

* `llm-as-function`能否按照既定格式返回信息主要取决于使用的模型性能，有时，LLM无法按照指定格式返回信息，从而抛出异常，你可以设置`parse_mode="accept_raw"`以获得模型返回的原始信息，从而避免抛出异常。

  代表你遇到rate limit限制了, 考虑进行换模型或者在每次使用function后sleep一段时间
