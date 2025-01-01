import pytest
from ollama import chat
from ollama import ChatResponse


def test_basic_chat():
    response: ChatResponse = chat(model='llama2', messages=[
        {
            'role': 'user',
            'content': 'Hello!',
        },
    ])

    assert isinstance(response, ChatResponse)
    assert response.message.content != ""
    assert isinstance(response.message.content, str)


def test_chat_multiple_messages():
    messages = [
        {'role': 'user', 'content': 'Hello!'},
        {'role': 'assistant', 'content': 'Hi there!'},
        {'role': 'user', 'content': 'How are you?'}
    ]
    response: ChatResponse = chat(model='llama2', messages=messages)

    assert isinstance(response, ChatResponse)
    assert response.message.content != ""


@pytest.mark.parametrize("model", ['llama2', 'llama3.1'])
def test_different_models(model):
    response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'user',
            'content': 'Hello!',
        },
    ])

    assert isinstance(response, ChatResponse)
    assert response.message.content != ""
