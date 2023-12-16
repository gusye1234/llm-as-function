import erniebot
from .utils import logger


async def ernie_single_acreate(query, model="ernie-bot-4"):
    response = await erniebot.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "user", "content": query}],
    )
    return response.get_result()


def ernie_single_create(query, model="ernie-bot-4", max_retry=3, temperature=0.1):
    retry = 0
    while retry < max_retry:
        try:
            response = erniebot.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                temperature=temperature,
            )
            return response.get_result()
        except erniebot.errors.APIError:
            retry += 1
            logger.warning(f"Failed {retry} times, retrying...")
    raise erniebot.errors.APIError("Max retry is reached")
