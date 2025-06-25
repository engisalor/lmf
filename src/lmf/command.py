"""Module with code for CLI commands."""

from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from langchain_core.load import dumpd


class Command:
    """Base class for CLI commands."""

    log_file: Path
    times: dict

    def add_dumpd(self, name: str, runnable: Any):
        """Attempts to dump any langchain runnable to a dict."""
        setattr(self, name, dumpd(runnable))

    def now(self, tag: str):
        """Gets current time via perf_counter()."""
        self.times[tag] = perf_counter()

    def duration(self):
        """Calculates the elapsed time between times["start"] and times["stop"]."""
        start = self.times.get("start")
        stop = self.times.get("stop")
        if start and stop:
            self.times["duration"] = round(stop - start, 3)

    def save_yaml(self):
        """Saves the command instance's parameters to a log file."""
        self.duration()
        print(f"... duration: {self.times.get('duration')}")
        with open(self.log_file.with_suffix(f".{self.run}.log.yml"), "w") as f:
            yaml.dump(
                vars(self), f, sort_keys=True, allow_unicode=True, encoding="utf-8"
            )

    def __repr__(self):
        return yaml.dump(vars(self), allow_unicode=True)
