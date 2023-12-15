from .llm_as_function import LLMFunc


def f(func):
    def new_func(*args, **kwargs):
        print("Here")
        return func(*args, **kwargs)

    return new_func


@f
def see(x):
    if x == 1 or x == 0:
        return x
    return see(x - 1) + see(x - 2)
