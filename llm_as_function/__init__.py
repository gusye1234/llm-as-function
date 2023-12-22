from .llm_func import LLMFunc, Final

ernie_func = LLMFunc(model="ernie-bot-4", temperature=0.1)
gpt35_func = LLMFunc(temperature=0.1)
gpt4_func = LLMFunc(temperature=0.1, model="gpt-4-1106-preview")

__author__ = "Jianbai Ye"
__version__ = "0.0.1"
__url__ = "https://github.com/gusye1234/llm-as-function"
