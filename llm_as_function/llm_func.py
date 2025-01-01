import inspect
import json
from copy import copy
from dataclasses import dataclass, field
from functools import wraps
import os
from typing import Literal

import ollama
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from ollama import Client as OllamaClient, AsyncClient as OllamaAsyncClient
from pydantic import BaseModel, ValidationError

from llm_as_function.types import LLMFuncConfig, RuntimeOptions, empty_runtime_options, Tool

from .errors import InvalidFunctionParameters, InvalidLLMResponse, ModelDoesNotSupportToolUse
from .fn_calling import function_to_name, get_argument_for_function, parse_function
from .models import (
    get_json_schema_prompt,
    openai_single_acreate,
    openai_single_create,
    ollama_single_acreate,
    ollama_single_create,
)
from .utils import LimitAPICalling, clean_output_parse, generate_schema_prompt, logger


def model_factory(model_name: str) -> Literal["openai", "ollama"]:
    OPENAI_STARTS_WITH = ["gpt"]
    OLLAMA_STARTS_WITH = ["llama", "qwq"]

    if any(model_name.startswith(prefix) for prefix in OPENAI_STARTS_WITH):
        return "openai"
    elif any(model_name.startswith(prefix) for prefix in OLLAMA_STARTS_WITH):
        return "ollama"
    raise NotImplementedError(f"llm-as-function currently supports OpenAI models, not {model_name}")


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

    parse_mode: Literal["error", "accept_raw"] = "error"
    output_schema: type[BaseModel] | None = None
    output_json: str | None = None  # The string generated from the output schema to be embedded into the prompt payload
    prompt_template: str = ""  # The actual prompt of the llmfunc i.e. core logic
    model: str = "gpt-3.5-turbo-1106"
    temperature: float = 0.1
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    ollama_base_url: str | None = None
    async_max_time: int | None = None
    has_tool_support: bool = False
    async_wait_time: float = 0.1
    runtime_options: RuntimeOptions = field(default_factory=empty_runtime_options)

    def __post_init__(self):
        assert self.parse_mode in ["error", "accept_raw",], f"Parse mode must in ['error', 'accept_raw'], not {self.parse_mode}"
        self.config: LLMFuncConfig = LLMFuncConfig(model=self.model, temperature=self.temperature, has_tool_support=self.has_tool_support)
        self.provider = model_factory(self.config["model"])

        self._bp_runtime_options = copy(self.runtime_options)
        self.fn_callings = {}
        self.async_models = {}

        if self.provider == "openai":
            if self.openai_api_key is None:
                logger.warning("OpenAI api key is not set, OPENAI_API_KEY env variable will be used instead")
                self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

            assert self.openai_api_key != "", "You must have OpenAI api key input, or set OPENAI_API_KEY in your environment."

            self.openai_client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
            self.openai_async_client = AsyncOpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)

        if self.provider == "ollama":
            if self.ollama_base_url is None:
                logger.warning("Ollama base url is not set, ollama will use default")

            self.ollama_client = OllamaClient(host=self.ollama_base_url)
            self.ollama_async_client = OllamaAsyncClient(host=self.ollama_base_url)

        if self.async_max_time is None:
            self.async_models["openai"] = openai_single_acreate
            self.async_models["ollama"] = ollama_single_acreate
        else:
            _limiter_decorator = LimitAPICalling(max_size=self.async_max_time, waiting_time=self.async_wait_time)
            self.async_models["openai"] = _limiter_decorator(openai_single_acreate)
            self.async_models["ollama"] = _limiter_decorator(ollama_single_acreate)

    def reset(self):
        """Reset the llmfuncs to the initial (default) state"""
        self.prompt_template = ""
        self.output_schema = None
        self.output_json = None
        self.runtime_options = copy(self._bp_runtime_options)
        self.func_callings = []
        self.fn_callings = {}

    def prompt(self, prompt_template: str):
        """Sets the llmfuncs prompt template"""
        self.prompt_template = prompt_template.strip("\n ")

        return self

    def func(self, func):
        """
        Adds function as a 'tool' to be used by the llms.

        Some LLM's support tool architecture, where you can define a function that the llm can 'use' during its
        responses.

        To learn more for OpenAI's api see:
            * https://community.openai.com/t/new-api-feature-forcing-function-calling-via-tool-choice-required/731488
            * https://platform.openai.com/docs/guides/function-calling/function-calling-behavior
            * https://platform.openai.com/docs/api-reference/chat/create

        These functions are expected to return a string, and the function arguments are expected to be a Pydantic BaseModel.
        i.e. Single argument function that returns a string.


        Some LLM's do not support tool architecture, and will raise an error (ModelDoesNotSupportToolUse) if you try to use this feature.
        """
        # MAYBE Rename to add_tool
        if not self.config["has_tool_support"]:
            raise ModelDoesNotSupportToolUse(self.config["model"])

        if self.provider not in ["openai", "ollama"]:
            raise NotImplementedError(f"Function calling for {self.provider} is not supported yet")

        self.fn_callings[function_to_name(func)] = func

        func_desc = parse_function(func)
        new_tool = Tool(type="function", function=func_desc)
        self.runtime_options["tools"].append(new_tool)
        # self.runtime_options["tool_choice"] = "auto" #  Already default

        return self

    def output(self, output_schema: type[BaseModel]):
        """
        Sets the llmfuncs output schema, also generates a text representation of the schema
        to be embedded into the prompt payload.

        """
        self.output_schema = output_schema
        self.output_json = generate_schema_prompt(output_schema)
        return self

    def parse_output(self, output: str, output_schema: type[BaseModel]) -> Final:
        """
        Cleans the output and parses it to the given output_schema.

        """
        logger.debug(f"Got output: {output}")
        json_str = clean_output_parse(output)

        if json_str is None:
            logger.error(f"Failed to parse output: {output}")
            if self.parse_mode == "error":
                raise InvalidLLMResponse(f"Failed to parse output to a valid json: {output}")
            elif self.parse_mode == "accept_raw":
                return Final(raw_response=output)
            raise InvalidLLMResponse(f"Failed to parse output: {output}")

        output_dict = output_schema(**json.loads(json_str)).model_dump()

        return Final(output_dict)

    def _init_setup(self, func):
        return_annotation = func.__annotations__.get("return", None)

        if return_annotation is not None:
            assert issubclass(return_annotation, BaseModel), "Return must be a Pydantic BaseModel"
            self.output(return_annotation)

        if self.output_schema is None and return_annotation is None:
            raise ValueError("You must specify the output schema or the function return annotation")

        if self.prompt_template == "":
            if func.__doc__ is None:
                raise ValueError("You must specify the prompt template or the function docstring")

            self.prompt(func.__doc__)

        return (
            self.prompt_template,
            self.output_json,
            self.output_schema,
            self.runtime_options,
            self.fn_callings,
        )

    def _fill_prompt(self, kwargs: dict, local_var: Final | dict, prompt_template: str) -> Final | str:
        """Fills the prompt template with the given kwargs and local_var, if local_var is a Final object, it will return the object"""
        if local_var is not None:
            if isinstance(local_var, Final):
                return local_var
            elif isinstance(local_var, dict):
                prompt = prompt_template.format(**kwargs, **local_var)
            else:
                raise NotImplementedError(f"UnSupported branch {type(local_var)}, please use one of the branch class: Final, dict")
        else:
            prompt = prompt_template.format(**kwargs)

        return prompt

    def _provider_response(self, prompt, runtime_options={}, fn_callings={}):
        logger.debug(runtime_options)

        if self.provider == "openai":
            chat_completion = openai_single_create(
                prompt,
                self.openai_client,
                runtime_options=runtime_options,
                model=self.config["model"],
                temperature=self.config["temperature"],
            )

            raw_result: ChatCompletionMessage = chat_completion.choices[0].message

            # If there is no tool_calls, return the content
            if raw_result.tool_calls is None:
                return raw_result.content

            # If there is tool_calls, call the functions
            return self._function_call_branch(prompt, raw_result, runtime_options, fn_callings)

        if self.provider == "ollama":
            chat_response = ollama_single_create(
                prompt,
                self.ollama_client,
                runtime_options=runtime_options,
                model=self.config["model"],
                temperature=self.config["temperature"],
            )

            raw_results = chat_response.message

            # If there is no tool_calls, return the content
            if raw_results.tool_calls is None:
                return raw_results.content

            # If there is tool_calls, call the functions
            return self._function_call_branch(prompt, raw_results, runtime_options, fn_callings)

        raise NotImplementedError(f"Provider [{self.provider}] is not supported yet")

    def _form_function_messages(
        self,
        tool_message: ChatCompletionMessage | ollama.Message,
        fn_callings={},
        history_messages=[],
    ):
        function_messages = history_messages + [tool_message]

        tool_calls = tool_message.tool_calls

        if tool_calls is None:
            raise ValueError("tool_calls is None")

        for tool_call in tool_calls:
            function_name = tool_call.function.name

            try:
                function_to_call = fn_callings[function_name]
            except KeyError as e:
                logger.error(f"function name is never added: {function_name}")
                raise e

            function_args_json = tool_call.function.arguments  # For ollama this is Mapping[str, Any] and for openai this is str (JSON)

            # Convert (or try to) ollama's Mapping[str, Any] to str
            if not isinstance(function_args_json, str):
                try:
                    function_args_json = json.dumps(function_args_json)
                except Exception as e:
                    raise ValueError(f"Failed to convert function_args_json to str: {function_args_json}. Failed with exception: {e}")

            logger.debug(f"Calling function {function_name} with args {function_args_json}")

            validate_type: type[BaseModel] = get_argument_for_function(function_to_call)

            try:
                function_args_parsed = validate_type.model_validate_json(function_args_json)
            except (ValueError, ValidationError):
                raise InvalidFunctionParameters(function_name, function_args_json)

            try:
                function_response = function_to_call(function_args_parsed)
            except Exception as e:
                logger.error(f"Occur error when running {function_name}")
                raise e

            assert isinstance(function_response, str), f"Expect function [{function_name}] to return str, not {type(function_response)}"

            message = {
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }

            # openai has id, ollama has tool_call_id
            if hasattr(tool_call, 'id'):
                assert not isinstance(tool_call, ollama.Message.ToolCall), "tool_call is not expected to be ollama.Message.ToolCall since it has no id"
                message["tool_call_id"] = tool_call.id

            function_messages.append(message)

        return function_messages

    def _function_call_branch(
        self,
        prompt,
        tool_message: ChatCompletionMessage | ollama.Message,
        runtime_options={},
        fn_callings={},
        history_messages=[],
    ):
        """Recursively call the functions in the tool_calls, each time appending the function response to funciton_messages and calling the next function"""
        function_messages = self._form_function_messages(tool_message, fn_callings, history_messages)

        logger.debug(f"Function message {function_messages}")

        if self.provider == "openai":
            chat_completion = openai_single_create(
                prompt,
                self.openai_client,
                runtime_options=runtime_options,
                function_messages=function_messages,
                model=self.config["model"],
                temperature=self.config["temperature"],
            )

            raw_result: ChatCompletionMessage = (chat_completion.choices[0].message)

            if raw_result.tool_calls is None:
                return raw_result.content

            return self._function_call_branch(prompt, raw_result, runtime_options, fn_callings, function_messages)

        if self.provider == "ollama":
            chat_response = ollama_single_create(
                prompt,
                self.ollama_client,
                runtime_options=runtime_options,
                function_messages=function_messages,
                model=self.config["model"],
                temperature=self.config["temperature"],
            )

            raw_results = chat_response.message

            if raw_results.tool_calls is None:
                return raw_results.content

            return self._function_call_branch(prompt, raw_results, runtime_options, fn_callings, function_messages)

        raise NotImplementedError(f"Function calling for provider [{self.provider}] is not supported yet")

    async def _provider_async_response(
        self, prompt, runtime_options={}, fn_callings={}
    ):
        if self.provider == "openai":
            chat_completion: ChatCompletion = await self.async_models["openai"](
                prompt,
                self.openai_async_client,
                runtime_options=runtime_options,
                model=self.config["model"],
                temperature=self.config["temperature"],
            )

            openai_raw_result: ChatCompletionMessage = chat_completion.choices[0].message

            if openai_raw_result.tool_calls is None:
                return openai_raw_result.content

            return await self._async_function_call_branch(prompt, openai_raw_result, runtime_options, fn_callings)

        if self.provider == "ollama":
            raw_result: ollama.ChatResponse = await self.async_models["ollama"](
                prompt,
                self.ollama_async_client,
                runtime_options=runtime_options,
                model=self.config["model"],
                temperature=self.config["temperature"],
            )

            ollama_raw_result: ollama.Message = raw_result.message

            if ollama_raw_result.tool_calls is None:
                return ollama_raw_result.content

            return await self._async_function_call_branch(prompt, ollama_raw_result, runtime_options, fn_callings)

        raise NotImplementedError(f"Provider [{self.provider}] is not supported yet")

    async def _async_function_call_branch(
        self,
        prompt,
        tool_message: ChatCompletionMessage | ollama.Message,
        runtime_options={},
        fn_callings={},
        history_messages=[],
    ):
        function_messages = self._form_function_messages(
            tool_message, fn_callings, history_messages
        )
        logger.debug(f"Function message {function_messages}")

        if self.provider == "openai":
            chat_completion: ChatCompletion = await self.async_models["openai"](
                prompt,
                self.openai_async_client,
                runtime_options=runtime_options,
                function_messages=function_messages,
                model=self.config["model"],
                temperature=self.config["temperature"],
            )

            openai_raw_result: ChatCompletionMessage = chat_completion.choices[0].message

            if openai_raw_result.tool_calls is None:
                return openai_raw_result.content

            return await self._async_function_call_branch(prompt, openai_raw_result, runtime_options, fn_callings, function_messages)

        if self.provider == "ollama":
            raw_result: ollama.ChatResponse = await self.async_models["ollama"](
                prompt,
                self.ollama_async_client,
                runtime_options=runtime_options,
                function_messages=function_messages,
                model=self.config["model"],
                temperature=self.config["temperature"],
            )

            ollama_raw_result: ollama.Message = raw_result.message

            if ollama_raw_result.tool_calls is None:
                return ollama_raw_result.content

            return await self._async_function_call_branch(prompt, ollama_raw_result, runtime_options, fn_callings, function_messages)

        raise NotImplementedError(
            f"Function calling for provider [{self.provider}] is not supported yet"
        )

    def _append_json_schema(self, prompt: str, output_json: str):
        """Gets the json schema prompt and appends it to the prompt to make models output json like the schema"""
        append_prompt = get_json_schema_prompt(self.provider, self.model).format(json_schema=output_json)
        return prompt + append_prompt

    def __call__(self, func):
        # parse input
        (
            prompt_template,
            output_json,
            output_schema,
            runtime_options,
            fn_callings,
        ) = self._init_setup(func)

        self.reset()

        @ wraps(func)
        def new_func(**kwargs):
            local_var = func(**kwargs)
            logger.debug(f"[Variables] function args:{kwargs}, local vars: {local_var}")

            prompt = self._fill_prompt(kwargs, local_var, prompt_template)

            if isinstance(prompt, Final):
                # Docs say "The `Final` is a class in `llm-as-function`, and returning this class indicates that you do not need the large model to process your output."
                # So this should just return
                return prompt

            if output_json is None:
                raise ValueError("The output_json is None when calling llmfunction. Most likely output_schema isn't supplied so the output_json couldn't be generated")

            prompt = self._append_json_schema(prompt, output_json)
            logger.debug(prompt)

            raw_result = self._provider_response(
                prompt, runtime_options=runtime_options, fn_callings=fn_callings
            )

            if not isinstance(raw_result, str):
                raise ValueError(f"Expected raw_result to be of type 'str' but it is of type '{type(raw_result)}'")

            if output_schema is None:
                raise ValueError("The output_schema is None, it is expected to be a type[BaseModel]")

            result = self.parse_output(raw_result, output_schema)

            return result

        return new_func

    def async_call(self, func):
        (
            prompt_template,
            output_json,
            output_schema,
            runtime_options,
            fn_callings,
        ) = self._init_setup(func)

        self.reset()

        @ wraps(func)
        async def new_func(**kwargs):
            if inspect.iscoroutinefunction(func):
                local_var = await func(**kwargs)
            else:
                local_var = func(**kwargs)
            logger.debug(f"[Variables] function args:{kwargs}, local vars: {local_var}")

            prompt = self._fill_prompt(kwargs, local_var, prompt_template)

            if isinstance(prompt, Final):
                # Docs say "The `Final` is a class in `llm-as-function`, and returning this class indicates that you do not need the large model to process your output."
                # So this should just return
                return prompt

            if output_json is None:
                raise ValueError("The output_json is None when calling llmfunction. Most likely output_schema isn't supplied so the output_json couldn't be generated")

            prompt = self._append_json_schema(prompt, output_json)
            logger.debug(prompt)

            raw_result = await self._provider_async_response(
                prompt, runtime_options=runtime_options, fn_callings=fn_callings
            )

            if not isinstance(raw_result, str):
                raise ValueError(f"Expected raw_result to be of type 'str' but it is of type '{type(raw_result)}'")

            if output_schema is None:
                raise ValueError("The output_schema is None, it is expected to be a type[BaseModel]")

            result = self.parse_output(raw_result, output_schema)
            logger.debug(f"Return {result}")

            return result

        return new_func

    def generate_llm_description(self, **kwargs):
        raise NotImplementedError
        # prompt = self.prompt_template.format(input_args)
