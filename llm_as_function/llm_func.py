import os
from enum import StrEnum
from pydantic import BaseModel
from dataclasses import dataclass
import json
from openai import OpenAI
from .utils import generate_schema_prompt, logger, clean_output_parse
from .models import ernie_single_create, openai_single_create

OUTPUT_JSON_PROMPT = """

你必须要按照如下的JSON格式代码输出你的结果, 否则你的结果将无法被评测系统正确解析:
{json_schema}
"""


def model_factory(model_name: str):
    if model_name.startswith("ernie"):
        return "ernie"
    if model_name.startswith("gpt"):
        return "openai"
    raise NotImplementedError(
        f"llm-as-function currently supports OpenAI or Ernie models, not {model_name}"
    )


@dataclass
class Final:
    pack: dict | None = None
    raw_response: str | None = None

    def ok(self):
        return self.pack is not None

    def unpack(self):
        if self.pack is not None:
            return self.pack
        return self.raw_response


@dataclass
class LLMFunc:
    """Use LLM as a function"""

    parse_mode: str = "error"
    output_schema: BaseModel | None = None
    output_json: dict | None = None
    prompt_template: str = ""
    model: str = "gpt-3.5-turbo-1106"
    temperature: float = 0.1
    deps: list = []
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    def __post_init__(self):
        assert self.parse_mode in [
            "error",
            "accept_raw",
        ], f"Parse mode must in ['error', 'accept_raw'], not {self.parse_mode}"
        self.config: dict = dict(model=self.model, temperature=self.temperature)
        self.provider = model_factory(self.config["model"])
        if self.provider == "openai":
            assert (
                self.openai_api_key != ""
            ), "You must have OpenAI api key input, or set OPENAI_API_KEY in your environment."
            self.openai_client = OpenAI(
                api_key=self.openai_api_key, base_url=self.openai_base_url
            )

    def reset(self):
        self.prompt_template = ""
        self.output_schema = None
        self.output_json = None

    def prompt(self, prompt_template):
        self.prompt_template = prompt_template.strip("\n ")

        return self

    def output(self, output_schema):
        self.output_schema = output_schema
        self.output_json = generate_schema_prompt(output_schema)
        return self

    def parse_output(self, output, output_schema):
        try:
            json_str = clean_output_parse(output)
            output = output_schema(**json.loads(json_str)).model_dump()
            return Final(output)
        except:
            logger.error(f"Failed to parse output: {output}")
            if self.parse_mode == "error":
                raise ValueError(f"Failed to parse output: {output}")
            elif self.parse_mode == "accept_raw":
                return Final(raw_response=output)
            raise ValueError(f"Failed to parse output: {output}")

    def __call__(self, func):
        # parse input
        return_annotation = func.__annotations__.get("return", None)
        if return_annotation is not None:
            assert issubclass(
                return_annotation, BaseModel
            ), "Return must be a Pydantic BaseModel"
            self.output(return_annotation)
        if self.output_schema is None and return_annotation is None:
            raise ValueError(
                "You must specify the output schema or the function return annotation"
            )
        if self.prompt_template == "":
            if func.__doc__ is None:
                raise ValueError(
                    "You must specify the prompt template or the function docstring"
                )
            self.prompt(func.__doc__)

        prompt_template = self.prompt_template
        output_json = self.output_json
        output_schema = self.output_schema

        self.reset()

        def new_func(**kwargs):
            local_var = func(**kwargs)
            # input_args = self.input_schema(**kwargs)
            logger.debug(f"{kwargs}, {local_var}")
            if local_var is not None:
                if isinstance(local_var, Final):
                    return local_var
                elif isinstance(local_var, dict):
                    prompt = prompt_template.format(**kwargs, **local_var)
                else:
                    raise NotImplementedError(
                        f"UnSupported branch {type(local_var)}, please use one of the branch class: Final, dict"
                    )
            else:
                prompt = prompt_template.format(**kwargs)
            if output_json is not None:
                prompt = prompt + OUTPUT_JSON_PROMPT.format(json_schema=output_json)

            logger.debug(prompt)

            if self.provider == "ernie":
                raw_result = ernie_single_create(prompt, **self.config)
            elif self.provider == "openai":
                raw_result = openai_single_create(
                    prompt, self.openai_client, **self.config
                )
            result = self.parse_output(raw_result, output_schema)
            logger.debug(f"Return {result}")

            return result

        return new_func

    def generate_llm_description(self, **kwargs):
        pass
        # prompt = self.prompt_template.format(input_args)

    def deps(self, llm_func):
        pass
