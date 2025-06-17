"""Module for read/write operations."""

import json
from ast import literal_eval
from pathlib import Path
from typing import List

import yaml
from langchain_core.prompt_values import ChatPromptValue, PromptValue

from clilm.utils import message_from_dict


class YamlLoader:
    """A class to manage loading and saving YAML files."""

    base_dir: str | Path

    @staticmethod
    def save_yaml(data: List[dict], file, base_dir=""):
        with open(base_dir / Path(file).with_suffix(".yml"), "w") as f:
            yaml.dump(data, f, allow_unicode=True, encoding="utf-8")

    @staticmethod
    def load_yaml(file, base_dir: str | Path = "") -> List[dict]:
        file = Path(file).with_suffix(".yml")
        default_filepath = base_dir / Path(file)
        if default_filepath.exists():
            file = default_filepath
        with open(file) as stream:
            return yaml.safe_load(stream)

    def __repr__(self):
        return f"{self.__class__}: base_dir = '{self.base_dir}'"


def prompts_to_yaml(prompts: List[PromptValue], file: Path | str | None):
    if isinstance(prompts, PromptValue):
        prompts = list(prompts)
    out = []
    for prompt in prompts:
        messages = prompt.to_messages()
        messages_dump = [x.model_dump_json(exclude_none=True) for x in messages]
        messages_dt = [json.loads(x) for x in messages_dump]
        messages_dt = [{k: v for k, v in x.items() if v} for x in messages_dt]
        for i, message in enumerate(messages_dt):
            if message["type"] == "ai":
                try:
                    messages_dt[i]["content"] = literal_eval(message["content"])
                except:
                    pass
        out.append(messages_dt)
    if file:
        with open(file, "w") as stream:
            yaml.dump(out, stream, allow_unicode=True, encoding="utf-8")
    else:
        return out


def prompts_from_yaml(file):
    with open(file, encoding="utf-8") as stream:
        ls = yaml.safe_load(stream)
    for i, prompt in enumerate(ls):
        for x, message in enumerate(prompt):
            ls[i][x]["content"] = str(message["content"])

    ls = [[message_from_dict(y) for y in x] for x in ls]
    return [ChatPromptValue(messages=x) for x in ls]
