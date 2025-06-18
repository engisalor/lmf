"""Module for preparing embedding recipes."""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from clilm.utils import OllamaEmbeddingsNormalized, make_custom_parameter

### classes for embeddings go here ###
# these must be EmbeddingsType subclasses


class EmbeddingsType:
    """Base class for embeddings recipes."""

    embeddings: HuggingFaceEmbeddings

    def get(self) -> HuggingFaceEmbeddings:
        """Returns initialized embeddings."""
        return self.embeddings


class HuggingFace(EmbeddingsType):
    """Recipe for HuggingFace embeddings."""

    def __init__(
        self,
        model="Qwen/Qwen3-Embedding-0.6B",
        encode_kwargs={"normalize_embeddings": True},
        **kwargs,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model=model,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )


class Ollama(EmbeddingsType):
    """Recipe for Ollama embeddings."""

    def __init__(
        self,
        model="qwen3-1.7b",
        **kwargs,
    ):
        self.embeddings = OllamaEmbeddings(
            model=model,
            **kwargs,
        )


class OllamaNorm(EmbeddingsType):
    """Recipe for Ollama embeddings, adding vector normalization."""

    def __init__(
        self,
        model="qwen3-1.7b",
        **kwargs,
    ):
        self.embeddings = OllamaEmbeddingsNormalized(
            model=model,
            **kwargs,
        )


### add new classes above this line ###
name = os.path.basename(__file__)
types = {
    k: v for k, v in locals().items() if getattr(v, "__base__", None) == EmbeddingsType
}
PARAMETER = make_custom_parameter(name, types)
