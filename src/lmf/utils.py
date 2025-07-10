"""Module for utility functions."""

import logging

import click
import numpy as np
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class BaseParam(click.ParamType):
    """Base class for custom parameter types."""

    def convert(self, value: str, param, ctx):
        """Converts the name of a custom object to the object itself."""
        dt = {k.lower(): v for k, v in self.types.items()}
        ls = list(self.types.keys())
        try:
            return dt[value.lower()]
        except KeyError:
            self.fail(f"{value!r} is not a recognized class in {ls}", param, ctx)


def make_custom_parameter(name, types):
    """Makes a custom parameter class and returns its instantiation."""
    obj = type(
        name,
        (BaseParam,),
        {
            "name": name,
            "types": types,
        },
    )
    return obj()


def message_from_dict(
    dict_message: dict,
    message_types={
        "system": SystemMessage,
        "human": HumanMessage,
        "ai": AIMessage,
    },
):
    """ "Returns a langchain message type (human, system, ai) from a dict."""
    return message_types[dict_message["type"]](**dict_message)


class OllamaEmbeddingsNormalized(OllamaEmbeddings):
    """Modified OllamaEmbeddings class that normalizes vectors.

    Source:
        https://github.com/ollama/ollama/issues/4128#issuecomment-2203403086
    """

    def _process_emb_response(self, input: str) -> list[float]:
        emb = super()._process_emb_response(input)
        return (np.array(emb) / np.linalg.norm(emb)).tolist()


def str_to_interval(s):
    """Returns a pandas Interval object from its string representation."""
    ls = [int(x) for x in s.strip("(]").split(", ")]
    return pd.Interval(*ls)


def camel_case(text: str) -> str:
    return text.title().replace(" ", "")[0].lower() + text.title().replace(" ", "")[1:]


def get_logger(name, level=logging.DEBUG, file=".lmf.log"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s.%(funcName)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
