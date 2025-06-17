"""Module for preparing embedding recipes."""

import os

from langchain_huggingface import HuggingFaceEmbeddings

from clilm.utils import make_custom_parameter

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
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        encode_kwargs={"normalize_embeddings": True},
        **kwargs,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model=embedding_model,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )


### add new classes above this line ###
name = os.path.basename(__file__)
types = {
    k: v for k, v in locals().items() if getattr(v, "__base__", None) == EmbeddingsType
}
PARAMETER = make_custom_parameter(name, types)
