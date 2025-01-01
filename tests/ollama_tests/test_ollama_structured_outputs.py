import pytest
from ollama import chat
from pydantic import BaseModel


class FriendInfo(BaseModel):
    name: str
    age: int
    is_available: bool


class FriendList(BaseModel):
    friends: list[FriendInfo]


def test_structured_output_parsing():
    response = chat(
        model='llama3.1',
        messages=[{
            'role': 'user',
            'content': 'I have two friends. The first is Ollama 22 years old busy saving the world, '
            'and the second is Alonso 23 years old and wants to hang out. Return a list of friends in JSON format'
        }],
        format=FriendList.model_json_schema(),
        options={'temperature': 0}
    )

    friends_response = FriendList.model_validate_json(response.message.content)  # type: ignore

    assert isinstance(friends_response, FriendList)
    assert len(friends_response.friends) == 2
    assert any(friend.name == "Ollama" for friend in friends_response.friends)
    assert any(friend.name == "Alonso" for friend in friends_response.friends)


def test_structured_output_validation():
    response = chat(
        model='llama3.1',
        messages=[{
            'role': 'user',
            'content': 'I have one friend named Bob who is 25 years old and is available. Return a list of friends in JSON format'
        }],
        format=FriendList.model_json_schema(),
        options={'temperature': 0}
    )

    friends_response = FriendList.model_validate_json(response.message.content)  # type: ignore

    assert isinstance(friends_response, FriendList)
    assert len(friends_response.friends) == 1

    friend = friends_response.friends[0]
    assert isinstance(friend.name, str)
    assert isinstance(friend.age, int)
    assert isinstance(friend.is_available, bool)
