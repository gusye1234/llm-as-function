from pydantic import BaseModel, Field
from llm_as_function import llama2_func
from llm_as_function.llm_func import Final

# This test is for testing ollamas structured output, llama2 on it's own canno't give JSON output most of the time
# But with the help of the structured output, we can get JSON output every time


class EmojiCharacter(BaseModel):
    emoji: str = Field(description="The emoji character")
    name: str = Field(description="Name of the emoji character")
    personality: str = Field(description="Personality traits of the emoji")


class EmojiStory(BaseModel):
    character1: EmojiCharacter
    character2: EmojiCharacter
    plot: str = Field(description="A short story about the interaction between the two emojis")
    moral: str = Field(description="The moral of the story")


@llama2_func
def generate_emoji_story(theme: str) -> EmojiStory:  # type: ignore
    """
    Create a short story about two emoji characters based on the given theme.
    The story should include their personalities and a moral lesson.
    """
    pass


def test_emoji_story_generation():
    response = generate_emoji_story(theme="friendship")

    assert isinstance(response, Final), f"Expected Final, but got {type(response)}"

    assert response.ok(), f"The response couldn't be parsed as JSON. Response: {response.raw_response}"

    result = EmojiStory(**response.unpack())  # type: ignore

    assert result.character1.emoji != "", "Expected character1 to have an emoji"
    assert result.character2.emoji != "", "Expected character2 to have an emoji"
    assert len(result.plot) > 0, "Expected plot to have some content"
    assert len(result.moral) > 0, "Expected moral to have some content"
