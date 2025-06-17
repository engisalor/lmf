"""Module for the `query` command, to make LLM queries."""

import os
from pathlib import Path

import click
import yaml
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass

from clilm import rate_limiter, schema
from clilm.command import Command
from clilm.io import YamlLoader, prompts_from_yaml

classes_dt = {
    "Output_structures": list(schema.PARAMETER.types.keys()),
    "Rate_limiters": list(rate_limiter.PARAMETER.types.keys()),
}
epilog = f"\b\nRECIPES\t*case insensitive*\n{yaml.dump(classes_dt)}"


class Query(Command):
    """Class to execute queries."""

    def __init__(
        self,
        ctx,
        model: str,
        output_structure: BaseModel,
        sample: int,
        temperature: float,
        think: bool,
        rate_limiter: rate_limiter.RateLimiterType,
    ):
        # input parameters
        self.ctx = ctx
        self.model = model
        self.output_structure = output_structure
        self.sample = sample
        self.temperature = temperature
        self.think = think
        self.rate_limiter = rate_limiter

        # explicit ctx
        self.base_dir: Path = ctx.obj["base_dir"]
        self.project_dir: Path = ctx.obj["project_dir"]
        self.prompt_dir: Path = ctx.obj["prompt_dir"]
        self.output_dir: Path = ctx.obj["output_dir"]
        self.log_dir: Path = ctx.obj["log_dir"]
        self.runs: int = ctx.obj["runs"]
        self.date: str = ctx.obj["date"]
        self.file_stem: Path = ctx.obj["file_stem"]

        # other parameters
        self.script = os.path.basename(__file__)
        self.run = 0
        self.times = {}
        self.warning = []
        self.prompt_file = self.prompt_dir / self.file_stem
        self.output_file = self.output_dir / self.file_stem
        self.log_file = self.log_dir / self.output_dir.name / self.file_stem
        self.chain_graph_file = self.log_dir / Path("mermaid-graph.md")

    def execute(self):
        "Run the queries."
        # load prompts
        self.prompts = prompts_from_yaml(
            self.prompt_file.with_suffix(f".{self.run}.yml")
        )

        # define LLM
        llm = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            rate_limiter=self.rate_limiter().get(),
        )

        # require structured output
        if getattr(self.output_structure, "to_json", None):
            llm = llm.with_structured_output(self.output_structure)
        self.llm_dump = self.add_dumpd("llm", llm)

        # execute
        if self.sample:
            prompts = self.prompts[: self.sample]
        else:
            prompts = self.prompts
        self.now("start")
        responses = llm.batch(prompts, think=self.think)
        self.now("stop")

        # save
        responses_json = [
            x.to_json() if self.output_structure else x for x in responses
        ]
        YamlLoader.save_yaml(
            responses_json, file=self.output_file.with_suffix(f".{self.run}.yml")
        )
        if self.run == 1:
            self.markdown_log(llm)
        self.save_yaml()

    def markdown_log(self, llm: Runnable):
        """Generate a markdown log file with a Mermaid chain graph."""
        with open(self.chain_graph_file, "w") as f:
            f.write(f"# {self.project_dir}\n\n{self.date}\n\n## chain graph\n\n")
            f.write(llm.get_graph().draw_mermaid())


@click.command(context_settings={"show_default": True}, epilog=epilog)
@click.option(
    "-m",
    "--model",
    default="qwen3:1.7b",
    help="Name of model (download models beforehand)",
)
@click.option(
    "-o",
    "--output-structure",
    default="Unstructured",
    type=schema.PARAMETER,
    help="A structured output class from schema.py",
)
@click.option(
    "--sample",
    default=0,
    help="Sample size (run first N prompts in a file; 0 == all)",
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.0,
    help="Model temperature",
)
@click.option("--think/--no-think", default=False, help="Toggle model thinking")
@click.option(
    "--rate-limiter",
    default="NoRateLimiter",
    type=rate_limiter.PARAMETER,
    help="A rate limiter class from rate_limiter.py",
)
@click.pass_context
def query(
    ctx,
    model: str,
    output_structure: ModelMetaclass,
    sample: int,
    temperature: float,
    think: bool,
    rate_limiter: rate_limiter.RateLimiterType,
):
    """Executes LLM final prompts given a model and output structure."""

    command = Query(
        ctx=ctx,
        model=model,
        output_structure=output_structure,
        sample=sample,
        temperature=temperature,
        think=think,
        rate_limiter=rate_limiter,
    )
    # execute runs
    for run in range(1, ctx.obj["runs"] + 1):
        click.echo(f"... {command.script} - run - {run}")
        command.run = run
        command.execute()


if __name__ == "__main__":
    query()
