from .llm_func import LLMFunc
import os

if "OPENAI_API_KEY" in os.environ:
    gpt35_func = LLMFunc(temperature=0.1)
    gpt4_func = LLMFunc(temperature=0.1, model="gpt-4o")

llama2_func = LLMFunc(temperature=0.1, model="llama2")
llama3_func = LLMFunc(temperature=0.1, model="llama3")

__author__ = "Jianbai Ye"
__version__ = "0.0.1"
__url__ = "https://github.com/gusye1234/llm-as-function"
