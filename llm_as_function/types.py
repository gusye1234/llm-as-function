from typing import List, Literal, Required, TypedDict


class Tool(TypedDict, total=False):
    function: Required[dict]
    type: Required[Literal["function"]]


class RuntimeOptions(TypedDict):
    tools: List[Tool]
    # For more info check https://platform.openai.com/docs/api-reference/chat/create
    tool_choice: Literal["none", "auto", "required"]


class LLMFuncConfig(TypedDict):
    model: Required[str]
    temperature: Required[float]
    has_tool_support: Required[bool]


def empty_runtime_options() -> RuntimeOptions:
    return {
        "tools": [],
        "tool_choice": "auto",
    }
