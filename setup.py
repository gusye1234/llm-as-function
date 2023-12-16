import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()


vars2find = ["__author__", "__version__", "__url__"]
varsllm = {}
with open("./llm_as_function/__init__.py") as f:
    for line in f.readlines():
        for v in vars2find:
            if line.startswith(v):
                line = line.replace(" ", "").replace('"', "").replace("'", "").strip()
                varsllm[v] = line.split("=")[1]

setuptools.setup(
    name="llm-as-function",
    url=varsllm["__url__"],
    version=varsllm["__version__"],
    author=varsllm["__author__"],
    description="Embed your LLM into a python function",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["llm_as_function"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=["rich", "erniebot", "pydantic"],
)
