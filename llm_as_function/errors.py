
class InvalidLLMResponse(Exception):
    pass


class InvalidFunctionParameters(Exception):
    """
    GPT passed invalid function parameters back to the caller

    """

    def __init__(self, invalid_function_name: str, invalid_parameters: str):
        super().__init__(f"Invalid function parameters: {invalid_parameters}")
        self.invalid_function_name = invalid_function_name
        self.invalid_parameters = invalid_parameters


class ModelDoesNotSupportToolUse(Exception):
    """
    The model does not support tool use

    """

    def __init__(self, model_name: str):
        super().__init__(f"The model {model_name} does not support tool use")
        self.model_name = model_name
