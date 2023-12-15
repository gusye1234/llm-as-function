import erniebot


async def ernie_single_acreate(query, model="ernie-bot-4"):
    response = await erniebot.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "user", "content": query}],
    )
    return response.get_result()


def ernie_single_create(query, model="ernie-bot-4"):
    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": query}],
    )
    return response.get_result()
