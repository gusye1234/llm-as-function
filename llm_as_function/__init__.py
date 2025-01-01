from .llm_func import LLMFunc, Final
import os

if "OPENAI_API_KEY" in os.environ:
    gpt35_func = LLMFunc(temperature=0.1, has_tool_support=True)
    gpt4_func = LLMFunc(temperature=0.1, model="gpt-4o", has_tool_support=True)

llama2_func = LLMFunc(temperature=0.1, model="llama2")
llama3_func = LLMFunc(temperature=0.1, model="llama3")
llama3_3_func = LLMFunc(temperature=0.1, model="llama3.3", has_tool_support=True)
qwq_func = LLMFunc(temperature=0.1, model="krtkygpta/qwq", has_tool_support=True)


__author__ = "Jianbai Ye"
__version__ = "0.0.1"
__url__ = "https://github.com/gusye1234/llm-as-function"
