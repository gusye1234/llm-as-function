import os
import sys
from dotenv import load_dotenv

sys.path.append("../")
load_dotenv()

import erniebot
from pydantic import BaseModel, Field
from llm_as_function import ernie_func, Final, LLMFunc
from llm_as_function.utils import logger
from rich import print
from rich.markdown import Markdown


erniebot.api_type = "aistudio"
erniebot.access_token = os.environ["ERNIE_KEY"]


class FollowingQuestionResult(BaseModel):
    answer: str = Field(description="用户问题的回答")
    following_questions: list[str] = Field(description="针对用户的问题, 提出一系列进一步完善这个问题的追问")


class Result(BaseModel):
    summary: str = Field(description="要点提取汇总")


CURRENT_Q = {}


@LLMFunc()
def ask_question(query) -> FollowingQuestionResult:
    """你是一个充满好奇的调研科学家.
    用户向你提了一个问题: {query}. 你需要根据你的知识简单回复用户的问题. 然后你需要发散你的思维, 思考如何提出更多的追问, 以便更好的完善这个问题."""
    pass


@LLMFunc()
def research(query, current_layer=1, max_layer=2) -> Result:
    """你是一个严谨的作者, 你需要理解用户的关于问题 {query} 的调研并且进行要点提取汇总
    用户现在已经进行了如下的调研:
    {answer}
    {sub_answer}
    """
    global CURRENT_Q
    if CURRENT_Q.get(f"layer_{current_layer}", None) is None:
        CURRENT_Q[f"layer_{current_layer}"] = [query]
    else:
        CURRENT_Q[f"layer_{current_layer}"].append(query)
    pack = ask_question(query=query).unpack()
    answer = pack["answer"]

    print(f"{query}: {answer}")

    following_questions = pack["following_questions"]
    sub_answers = []

    if current_layer >= max_layer:
        return Final({"summary": answer})

    for question in following_questions[:3]:
        print(f"{query} -> {question}")
        sub_answers.append(
            (
                question,
                research(
                    query=question, current_layer=current_layer + 1, max_layer=max_layer
                ).unpack()["summary"],
            )
        )

    sub_answer = "\n".join(
        [
            f"{'#'*(current_layer+1)} {question}\n{answer}"
            for question, answer in sub_answers
        ]
    )
    return Final({"summary": f"{'#'*current_layer} {query}\n{answer}\n{sub_answer}"})


import gradio as gr


def get_result(query, max_layer=2):
    message = research(query=query, max_layer=max_layer).unpack()["summary"]
    return gr.Markdown(message)


with gr.Blocks() as demo:
    gr.Markdown(
        """
# Knowledge Agent
Ask any question, I will discover all the dimensions of it
    """
    )
    with gr.Row(equal_height=False):
        with gr.Column():
            gr.Interface(
                fn=get_result,
                inputs=["text", "number"],
                outputs="markdown",
                allow_flagging="never",
            )
        with gr.Column():
            label = gr.JSON({})
            with gr.Row():
                refresh = gr.Button("Refresh")
                clean = gr.Button("Clean")

            @refresh.click(outputs=label)
            def fresh():
                return CURRENT_Q

            @clean.click(outputs=label)
            def fresh():
                global CURRENT_Q
                CURRENT_Q = {}
                return CURRENT_Q


demo.launch()
