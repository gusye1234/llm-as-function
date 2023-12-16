from pydantic import BaseModel
from dataclasses import dataclass
import json
from .utils import generate_schema_prompt, logger, clean_output_parse_ernie
from .models import ernie_single_create, ernie_single_acreate

OUTPUT_JSON_PROMPT = """

你必须要按照如下的JSON格式代码输出你的结果, 否则你的结果将无法被评测系统正确解析:
{json_schema}
"""


def model_factory(model_name: str):
    if model_name.startswith("ernie"):
        return "ernie"
    if model_name.startswith("gpt"):
        return "openai"


@dataclass
class Final:
    pack: dict

    def unpack(self):
        return self.pack


@dataclass
class LLMFunc:
    output_schema: BaseModel | None = None
    output_json: dict | None = None
    prompt_template: str = ""
    model: str = "ernie-bot-4"
    temperature: float = 0.1
    deps: list = []

    def __post_init__(self):
        self.config: dict = dict(model=self.model, temperature=self.temperature)
        self.provider = model_factory(self.config["model"])

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
            if self.provider == "ernie":
                json_str = clean_output_parse_ernie(output)
            output = output_schema(**json.loads(json_str)).model_dump()
        except:
            logger.error(f"Failed to parse output: {output}")
            raise ValueError(f"Failed to parse output: {output}")
        return output

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
                    return local_var.unpack()
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

            result = self.parse_output(raw_result, output_schema)
            logger.debug(f"Return {result}")

            return result

        return new_func

    def generate_llm_description(self, **kwargs):
        pass
        # prompt = self.prompt_template.format(input_args)

    def deps(self, llm_func):
        pass
