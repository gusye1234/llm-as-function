<div align="center">
  <h1>llm-as-function</h1>
  <p><strong>Embed LLM into your python function</strong></p>
  <p>
        <a href="https://pypi.org/project/llm-as-function/">
      <img src="https://img.shields.io/pypi/v/llm-as-function.svg">
    </a>
  </p>
</div>


`llm-as-function` is a Python library that helps you quickly build functions based on Large Language Models (LLMs). You can use `LLMFunc` as a decorator for your functions, while also providing type annotations and writing docstrings for your functions. `llm-as-function` will automatically complete the parameter filling by invoking the large model and return formatted output.

> `llm-as-function` 是一个帮助你快速构建基于LLM的函数的Python库. 你可以使用`LLMFunc`作为你函数的装饰器, 同时给你的函数进行类型标注和docstring的编写, `llm-as-function`会自动的通过调用大模型完成参数的填入, 并且返回格式化的输出. 

The `llm-as-function` also supports defining code bodies within the LLM function for precise inference control and business logic.

> `llm-as-function`还支持在LLM函数中定义代码体, 用于精确的推理控制和业务逻辑.

## Get Started
```
pip install llm-as-function
```
## Features

Basic usage: 

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

More demos in `examples/`

## Docs

`LLMFunc`

```python
# LLMFunc currently support OpenAI provider, Ernie SDK(文心一言)
@LLMFunc(model="gpt-3.5-turbo-1106", temperature=0.3, openai_base_url=..., openai_api_key=...)
def fool() -> Result:
    ...
@LLMFunc(model="ernie-bot-4", temperature=0.3)
def fool() -> Result:
    ...
    
# For your convenience, llm-as-function already instantiated some LLMFunc
from llm_as_function import gpt35_func, gpt4_func, ernie_funcx

@gpt35_func
def fool() -> Result:
    ...
    
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

* 当`llm-as-function`使用的是`ernie-bot-4`, 其API的访问对于rate limit限制的比较狠, 如果你遇到如下的Error

  ```
  erniebot.errors.APIError: Max retry is reached
  ```

  代表你遇到rate limit限制了, 考虑进行换模型或者在每次使用function后sleep一段时间
