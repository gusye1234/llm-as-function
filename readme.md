<div align="center">
  <h1>llm-as-function</h1>
  <p><strong>Embed your LLM into a python function</strong></p>
  <p>
        <a href="https://pypi.org/project/llm-as-function/">
      <img src="https://img.shields.io/pypi/v/llm-as-function.svg">
    </a>
  </p>
</div>


`llm-as-function` 是一个帮助你快速构建基于LLM的函数的Python库. 你可以使用`LLMFunc`作为你函数的装饰器, 同时给你的函数进行类型标注和docstring的编写, `llm-as-function`会自动的通过调用大模型完成参数的填入, 并且返回格式化的输出.

## Get Started
要使用llm_as_function代码库，首先需要安装它。可以按照以下步骤进行安装：
```
pip install llm-as-function
```
## Features

安装完成后, 可以如下使用

```python
import erniebot
from llm_as_function import LLMFunc
from pydantic import BaseModel, Field

# llm-as-function默认的LLM是百度的Ernie bot, 你需要获取他的API token才能使用
erniebot.api_type = "aistudio"
erniebot.access_token = os.environ["ERNIE_KEY"]

# 通过pydantic完成输入类型的验证和定义
class Result(BaseModel):
    emoji: str = Field(description="随机输出一个emoji")

# 使用装饰器, LLMFunc会自动识别你的函数的输入和输出, 以及函数的docstring.
# 在这里, 函数的DocString就是你的Prompt, 请谨慎设计
@LLMFunc()
def fool() -> Result:
    """
    你需要随机输出一个emoji
    """
    pass
  
print(foo()) # {emoji: "😅"}
```

你也可以动态的从在prompt中插入变量

```
@LLMFunc()
def fool2(emotion) -> Result:
    """
    你需要随机输出一个emoji, 我希望感情是{emotion}
    """
    pass
  
print(foo2(emotion="开心")) # {'emoji': '😊'}
```

经一步的, 你可以构建更加复杂的输出逻辑

```python
# ernie_func是一个已经实例化好的, 默认的装饰器
from llm_as_function import ernie_func 

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
```

你能拿到如下的`dict`

```python
{
    'emoji': {
        'emoji': '😄',
        'why':
'这个emoji表示开心和微笑的情绪，它传达出一种积极、友好的氛围。',
        'more': {
            'where':
'你可以在各种社交场合使用这个emoji，比如和朋友聊天、在社交媒体上评论或回复
别人的帖子，甚至在工作中与同事交流时也可以使用它来表示友善和亲切。',
            'warning':
'尽管这个emoji通常被看作是积极和友好的，但在某些文化和语境下，它可能被视
为不够真诚或过于简单。因此，在使用它时，你应该考虑到你的受众和上下文环境，以确
保你的表达是恰当和得体的。'
        }
    }
}
```

**最关键的**, 你可以在你的function当中插入`python`的代码, 他会在实际的LLM运行前运行, 所以你可以完成类似的事情:

```python
@ernie_func
def fool() -> Result:
    """
    你需要随机输出一个emoji
    """
		print("Logging once")
```

更有意思的是, 你可以当中调用别的函数, 别的LLM function (参考`examples/3_fibonacci.py`):

```python
from llm_as_function import LLMFunc, Final
class Result(BaseModel):
    value: int = Field(description="斐波那契数列计算的值")


@LLMFunc()
def f(x: int) -> Result:
    """
    你需要计算斐波那契数列的第{x}项, 你有他的前两项的值, 分别是{a}和{b}. 你计算第{x}项的方式是将前两项的值相加. 请你计算出第{x}项的值"""
    if x == 1 or x == 0:
      	# Final是llm-as-function的一个类, 返回这个类代表你不需要大模型处理你的这个输出. Final当中应该是一个dict, 他的格式和你定义的Result是相同的
        return Final({"value": x})
    a = f(x=x - 1)
    b = f(x=x - 2)
    # 正常的函数返回代表你向大模型传递了‘本地变量’, 你返回的变量会被插入到你的prompt当中.
    return {"a": a["value"], "b": b["value"]}

print(f(3)) # {value: 2}
```

更多的例子请参考 `examples/`



## API

```
# LLMFunc目前只支持ernie bot相关的模型
@LLMFunc(model="ernie-bot-4", temperature-0.3)
def fool() -> Result:
		...
```



## FQA

* `llm-as-function`的格式化返回依赖于你所使用模型的能力, 有时候大模型不一定能返回可解析的JSON格式, 进而导致Error

* `llm-as-function`默认使用的是`ernie-bot-4`, 其API的访问对于rate limit限制的比较狠, 如果你遇到如下的Error

  ```
  erniebot.errors.APIError: Max retry is reached
  ```

  代表你遇到rate limit限制了, 考虑进行换模型或者在每次使用function后sleep一段时间
