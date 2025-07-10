"""Module for the `query` command, to make LLM queries."""

import gc
import os
from pathlib import Path

import click
import torch
import yaml
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass

from lmf import chat_model, parse, rate_limiter, schema
from lmf.command import Command
from lmf.io import YamlLoader, prompts_from_yaml
from lmf.utils import get_logger

logger = get_logger(__name__)


classes_dt = {
    "Chat_models": list(chat_model.PARAMETER.types.keys()),
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
        chat_model: chat_model.ChatModelType,
        output_structure: BaseModel,
        sample: int,
        temperature: float,
        timeout: int,
        max_tokens: int,
        think: bool,
        rate_limiter: rate_limiter.RateLimiterType,
    ):
        # input parameters
        self.ctx = ctx
        self.model = model
        self.chat_model = chat_model
        self.output_structure = output_structure
        self.sample = sample
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
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
        self.seed: str = ctx.obj["seed"]

        # other parameters
        self.script = os.path.basename(__file__)
        self.run = 0
        self.times = {}
        self.warning = []
        self.prompt_file = self.prompt_dir / self.file_stem
        self.output_file = self.output_dir / self.file_stem
        self.log_file = self.log_dir / self.output_dir.name / self.file_stem
        self.chain_graph_file = self.log_dir / Path("mermaid-graph.md")
        self.gold_standard_file = self.project_dir / Path("gold.jsonl")

        # clear gpu memory
        torch.cuda.empty_cache()
        gc.collect()

    def execute(self):
        "Run the queries."
        self.now("start")
        # load prompts
        self.prompts = prompts_from_yaml(
            self.prompt_file.with_suffix(f".{self.run}.yml")
        )

        # define LLM
        llm: Runnable = self.chat_model(
            model=self.model,
            temperature=self.temperature,
            rate_limiter=self.rate_limiter().get(),
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            seed=self.seed,
        ).get()

        # require structured output
        if getattr(self.output_structure, "model_dump", None):
            llm = llm.with_structured_output(self.output_structure)
        self.llm_dump = self.add_dumpd("llm", llm)
        # execute
        if self.sample:
            prompts = self.prompts[: self.sample]
        else:
            prompts = self.prompts
        responses = llm.batch(prompts, think=self.think)
        # use structured output parser
        if self.output_structure and self.output_structure != schema.Unstructured:
            structure = self.output_structure.__name__
            parser: parse.Parser = getattr(parse, structure + "Parser")
            parser(self, responses)
        # or save to yaml
        else:
            out = self.output_file.with_suffix(f".{self.run}.yml")
            YamlLoader.save_yaml(responses, file=out)

        if self.run == 1:
            self.markdown_log(llm)
        self.now("stop")
        self.save_yaml()

        # clear gpu memory
        del llm
        torch.cuda.empty_cache()
        gc.collect()

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
    "--chat_model",
    default="Ollama",
    type=chat_model.PARAMETER,
    help="A chat model chat model class from chat.py",
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
    help="Model temperature (0.0 = more deterministic / 1.0 = more variable)",
)
@click.option(
    "--timeout",
    default=300,
    help="Response timeout (for cloud providers)",
)
@click.option(
    "--max_tokens",
    default=10000,
    help="Model maximum tokens per response",
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
    chat_model: chat_model.ChatModelType,
    output_structure: ModelMetaclass,
    sample: int,
    temperature: float,
    timeout: int,
    max_tokens: int,
    think: bool,
    rate_limiter: rate_limiter.RateLimiterType,
):
    """Executes LLM final prompts with a model, model provider and output structure."""
    command = Query(
        ctx=ctx,
        model=model,
        chat_model=chat_model,
        output_structure=output_structure,
        sample=sample,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        think=think,
        rate_limiter=rate_limiter,
    )
    # execute runs
    for run in range(1, ctx.obj["runs"] + 1):
        logger.info(f"{run}")
        command.run = run
        command.execute()
