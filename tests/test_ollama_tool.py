import pytest
from ollama import chat
from ollama import ChatResponse


@pytest.fixture
def math_tools():
    def add_two_numbers(a: int, b: int) -> int:
        return a + b

    def subtract_two_numbers(a: int, b: int) -> int:
        return a - b

    subtract_two_numbers_tool = {
        'type': 'function',
        'function': {
            'name': 'subtract_two_numbers',
            'description': 'Subtract two numbers',
            'parameters': {
                'type': 'object',
                'required': ['a', 'b'],
                'properties': {
                    'a': {'type': 'integer', 'description': 'The first number'},
                    'b': {'type': 'integer', 'description': 'The second number'},
                },
            },
        },
    }

    return {
        'tools': [add_two_numbers, subtract_two_numbers_tool],
        'functions': {
            'add_two_numbers': add_two_numbers,
            'subtract_two_numbers': subtract_two_numbers,
        }
    }


def test_tool_calls(math_tools):
    messages = [{'role': 'user', 'content': 'What is three plus one?'}]
    response: ChatResponse = chat(
        'llama3.1',
        messages=messages,
        tools=math_tools['tools']
    )

    assert response.message.tool_calls is not None
    assert len(response.message.tool_calls) > 0

    tool = response.message.tool_calls[0]
    function = math_tools['functions'].get(tool.function.name)
    assert function is not None

    result = function(**tool.function.arguments)
    assert isinstance(result, int)


def test_tool_conversation_flow(math_tools):
    messages = [{'role': 'user', 'content': 'What is five minus two?'}]
    response: ChatResponse = chat(
        'llama3.1',
        messages=messages,
        tools=math_tools['tools']
    )

    assert response.message.tool_calls is not None

    for tool in response.message.tool_calls:
        function = math_tools['functions'].get(tool.function.name)
        assert function is not None
        output = function(**tool.function.arguments)

        messages.append(response.message)  # type: ignore
        messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

    final_response = chat('llama3.1', messages=messages)
    assert isinstance(final_response.message.content, str)
    assert final_response.message.content != ""
