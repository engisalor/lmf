"""Module for preparing chat model chat model recipes."""

import os

from langchain_ollama import ChatOllama

from clilm.utils import make_custom_parameter

### classes for chat model chat models go here ###
# these must be ChatModelType subclasses


class ChatModelType:
    """Base class for chat model chat model recipes."""

    chat_model: ChatOllama

    def get(self) -> ChatOllama:
        """Returns initialized chat model."""
        return self.chat_model


class Ollama(ChatModelType):
    """Recipe for Ollama chat model chat model."""

    def __init__(
        self,
        model="qwen3:1.7b",
        temperature=0,
        rate_limiter=None,
        timeout=600,
        max_tokens=10000,
        **kwargs,
    ):
        self.chat_model = ChatOllama(
            model=model,
            temperature=temperature,
            rate_limiter=rate_limiter,
            timeout=timeout,
            max_tokens=max_tokens,
            **kwargs,
        )


### add new classes above this line ###
name = os.path.basename(__file__)
types = {
    k: v for k, v in locals().items() if getattr(v, "__base__", None) == ChatModelType
}
PARAMETER = make_custom_parameter(name, types)
