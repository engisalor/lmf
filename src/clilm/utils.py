"""Module for utility functions."""

import os

import click
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
