import os
import inspect
import asyncio
from functools import wraps
from types import UnionType
from pydantic import BaseModel
from re import DOTALL, finditer
from typing import List, Literal, Type, get_args, get_origin

import logging
from rich.logging import RichHandler

logger = logging.getLogger("agent")
if not logger.handlers:
    logger.addHandler(RichHandler())
    logger.setLevel(os.environ.get("LEVEL", "INFO"))


def find_json_response(full_response, extract_type=dict):
    """
    Takes a full response that might contain other strings and attempts to extract the JSON payload.
    Has support for truncated JSON where the JSON begins but the token window ends before the json is
    is properly closed.

    """
    # Deal with fully included responses as well as truncated responses that only have one
    if extract_type == dict:
        extracted_responses = list(
            finditer(r"({[^}]*$|{.*})", full_response, flags=DOTALL)
        )
    else:
        raise ValueError("Unknown extract_type")

    if not extracted_responses:
        print(
            f"Unable to find any responses of the matching type `{extract_type}`: `{full_response}`"
        )
        return None

    if len(extracted_responses) > 1:
        print("Unexpected response > 1, continuing anyway...", extracted_responses)

    extracted_response = extracted_responses[0]

    extracted_str = extracted_response.group(0)

    return extracted_str


def clean_output_parse(llm_output: str):
    return find_json_response(llm_output.strip())


def generate_schema_prompt(schema: Type[BaseModel]) -> str:
    """
    Converts the pydantic schema into a text representation that can be embedded
    into the prompt payload.

    """

    def generate_payload(model: Type[BaseModel]):
        payload = []
        for key, value in model.model_fields.items():
            field_annotation = value.annotation
            annotation_origin = get_origin(field_annotation)
            annotation_arguments = get_args(field_annotation)

            if field_annotation is None:
                continue
            elif annotation_origin in {list, List}:
                if issubclass(annotation_arguments[0], BaseModel):
                    payload.append(
                        f'"{key}": {generate_payload(annotation_arguments[0])}[]'
                    )
                else:
                    payload.append(f'"{key}": {annotation_arguments[0].__name__}[]')
            elif annotation_origin == UnionType:
                payload.append(
                    f'"{key}": {" | ".join([arg.__name__.lower() for arg in annotation_arguments])}'
                )
            elif annotation_origin == Literal:
                allowed_values = [f'"{arg}"' for arg in annotation_arguments]
                payload.append(f'"{key}": {" | ".join(allowed_values)}')
            elif issubclass(field_annotation, BaseModel):
                payload.append(f'"{key}": {generate_payload(field_annotation)}')
            else:
                payload.append(f'"{key}": {field_annotation.__name__.lower()}')
            if value.description:
                payload[-1] += f" // {value.description}"
        return "{\n" + ",\n".join(payload) + "\n}"

    return generate_payload(schema)


class LimitAPICalling:
    """The restriction for accessing the async GPT(acreate) is that only a maximum of max_size GPTs can be accessed at the same time."""

    def __init__(self, max_size=8, waiting_time=0.1) -> None:
        self.max_size = max_size
        self._current_bin = 0
        self.waiting_time = waiting_time

    def __call__(self, func):
        assert inspect.iscoroutinefunction(func), "func must be a coroutine function"

        @wraps(func)
        async def wait_func(*args, **kwargs):
            while True:
                if self._current_bin < self.max_size:
                    self._current_bin += 1
                    break
                await asyncio.sleep(self.waiting_time)
            logger.debug(f"Calling LLM [{self._current_bin}]/[{self.max_size}]")
            result = await func(*args, **kwargs)
            self._current_bin -= 1
            return result

        return wait_func
