from .llm_func import LLMFunc, Final
import os

if "OPENAI_API_KEY" in os.environ:
    gpt35_func = LLMFunc(temperature=0.1, has_tool_support=True)
    gpt4_func = LLMFunc(temperature=0.1, model="gpt-4o", has_tool_support=True)

llama2_func = LLMFunc(temperature=0.1, model="llama2", has_structured_output=True)
llama3_func = LLMFunc(temperature=0.1, model="llama3", has_structured_output=True)
llama3_1_func = LLMFunc(temperature=0.1, model="llama3.1", has_tool_support=True, has_structured_output=True)
llama3_3_func = LLMFunc(temperature=0.1, model="llama3.3", has_tool_support=True, has_structured_output=True)
llama3_2_1b_func = LLMFunc(temperature=0.1, model="llama3.2.1b", has_tool_support=True, has_structured_output=True)
qwq_func = LLMFunc(temperature=0.1, model="krtkygpta/qwq", has_tool_support=True, has_structured_output=True)
qwen2_1_5b_func = LLMFunc(temperature=0.1, model="qwen2:1.5b", has_tool_support=True, has_structured_output=True)  # This has tool support but i haven't be able to relabily use it

__author__ = "Jianbai Ye"
__version__ = "0.0.1"
__url__ = "https://github.com/gusye1234/llm-as-function"
