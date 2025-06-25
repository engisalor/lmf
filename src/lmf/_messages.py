"""A deprecated module for managing langchain messages."""

import json
from dataclasses import dataclass
from hashlib import blake2b
from typing import List

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.chat import ChatMessage


@dataclass
class MessageScope:
    """Predefined scopes for a message."""

    task: str = "A task that the system should perform."
    note: str = "A clarifying note for system behavior."

    def all(self):
        return vars(self).keys()


@dataclass
class MessageClasses:
    """Some message classes that loaded dicts can be converted into."""

    ai: BaseMessage = AIMessage
    human: BaseMessage = HumanMessage
    system: BaseMessage = SystemMessage
    chat: BaseMessage = ChatMessage

    def all(self):
        return vars(self)


@dataclass
class Messages:
    """Messages loaded from a YAML file."""

    messages: List[dict]
    classes = MessageClasses().all()

    def _as_message(self, message_type, **kwargs):
        return [
            self.classes[message_type](
                content=x.get("content", None),
                **x.get("kwargs", {}),
                **kwargs,
            )
            for x in self.messages
        ]

    def list(self):
        return self.messages

    def as_chat(self, role, **kwargs):
        kwargs |= {"role": role}
        return self._as_message("chat", **kwargs)

    def as_ai(self, **kwargs):
        return self._as_message("ai", **kwargs)

    def as_system(self, **kwargs):
        return self._as_message("system", **kwargs)

    def as_human(self, **kwargs):
        return self._as_message("human", **kwargs)

    def validate(self):
        scopes = MessageScope().all()
        for i, message in enumerate(self.messages):
            suffix = f"- {i}: {message}"
            if not "scope" in message.keys():
                raise ValueError(f"scope missing {suffix}")
            if not message["scope"] in scopes:
                m = f"unknown scope {suffix} - must be in {scopes}"
                raise ValueError(m)
            if not "content" in message.keys():
                raise ValueError(f"content missing {suffix}")
            if not message["scope"].strip():
                raise ValueError(f"scope is empty {suffix}")
            if not message["content"].strip():
                raise ValueError(f"content is empty {suffix}")
            if "kwargs" in message.keys():
                if message["kwargs"] is None:
                    self.messages[i]["kwargs"] = {}
                if not isinstance(message["kwargs"], dict):
                    raise ValueError(f"type(kwargs) != dict {suffix}")


class MessageHandler:
    """Class for handling messages."""

    @staticmethod
    def unique(messages: List[dict]):
        """Return unique messages"""
        return list({v["hash"]: v for v in messages}.values())

    def all(self):
        """Return all messages"""
        ls = self.messages
        return Messages(self.unique(ls))

    def find_any(self, **kwargs):
        """Finds messages matching any argument.

        kwargs: key-value pairs to search for. Values may be either a single value
            (`summarizer`) or a list (`["summarizer", "relation-extractor"]`)

        Examples:
            ```
            # find any matching values
            # (at least one match in these two k:v pairs)
            messages.find_any(
                file="summarizer",
                scope="note",
                )

            # find any matching values from a list
            # (at least one match in these four k:v pairs)
            messages.find_any(
                file=["summarizer", "relation-extractor"],
                scope=["task", "note"],
                )

            ```
        """
        dt = {k: list(v) for k, v in kwargs.items()}
        ls = []
        for k, v in dt.items():
            for i in v:
                ls.extend([x for x in self.messages if x.get(k, None) == i])
        return Messages(self.unique(ls))

    def find(self, **kwargs):
        """Finds messages matching every argument.

        kwargs: key-value pairs to search for. Only single values are accepted.
            (See `messages.find_any()` for more flexible matching.)

        Examples:
            ```
            # find all matching values
            messages.find_any(
                file="summarizer",
                scope="note",
                )
        """
        ls = []
        for message in self.messages:
            matches = []
            for k, v in kwargs.items():
                if message.get(k, None) == v:
                    matches.append(True)
                else:
                    matches.append(False)
            if set(matches) == set([True]):
                ls.append(message)
        return Messages(self.unique(ls))

    def __repr__(self):
        return yaml.dump(self.all(), allow_unicode=True, encoding="utf-8").replace(
            "\n\n", "\n"
        )

    def __init__(
        self,
        messages: list | None = None,
    ):
        Messages(messages).validate()
        for message in messages:
            _json = json.dumps(message, sort_keys=True)
            _hash = blake2b(bytes(_json, encoding="utf-8")).digest().hex()[:7]
            message["hash"] = _hash
        self.messages = messages
