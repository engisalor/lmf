"""Module for the `query` command, to make LLM queries."""

import gc
import importlib.util
import os
import sys
from pathlib import Path

import click
import torch
import yaml
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass

from lmf import chat_model, parse, rate_limiter, schema
from lmf.command import Command
from lmf.io import YamlLoader, prompts_from_yaml
from lmf.utils import get_logger, timer

logger = get_logger(__name__)


classes_dt = {
    "Chat_models": list(chat_model.PARAMETER.types.keys()),
    "Output_structures": list(schema.PARAMETER.types.keys()),
    "Rate_limiters": list(rate_limiter.PARAMETER.types.keys()),
}
epilog = f"\b\nRECIPES\t*case insensitive*\n{yaml.dump(classes_dt)}"


class Query(Command):
    """Class to execute queries."""

    def import_chain(self):
        logger.info(f"using {self.chain_file}")
        spec = importlib.util.spec_from_file_location(
            "lmf.project_chain", self.chain_file
        )
        chain_module = importlib.util.module_from_spec(spec)
        sys.modules["lmf.project_chain"] = chain_module
        spec.loader.exec_module(chain_module)
        self.chain_module = chain_module
        structures = [
            x
            for x in vars(self.chain_module).values()
            if getattr(x, "__base__", None) == BaseModel
        ]
        structures = [x for x in structures if not x.__name__.startswith("_")]
        if not structures:
            raise ValueError(f"no public pydantic classes in {self.chain_module}")
        self.output_structure = structures
        self.chain_sequential = getattr(self.chain_module, "sequential", False)

    def parse_output(
        self, responses: list[BaseModel], structure: BaseModel
    ) -> parse.Parser:
        parser: parse.Parser
        if self.chain_module:
            parser = getattr(self.chain_module, structure.__name__ + "Parser", None)
        else:
            parser = getattr(parse, structure.__name__ + "Parser", None)
        if not parser:
            parser = parse.YamlParser
        parser(command=self, responses=responses)

    @timer(logger=logger)
    def run_call(self, structure, i=0):
        llm: Runnable = self.llm
        if structure != schema.Unstructured:
            llm = llm.with_structured_output(structure)
        responses = llm.batch(self.prompts, think=self.think)
        if len(self.output_structure) > 1:
            self.yaml_out = self.output_dir / Path(
                f"{structure.__name__}.{self.run}.yml"
            )
        logger.info(f"{i} - {structure.__name__}")
        return responses

    def append_chat_history(self, responses: list[BaseMessage]):
        for p, r in zip(self.prompts, responses):
            p.messages.append(AIMessage(content=r.model_dump_json()))

    @timer(logger=logger)
    def run_chain_sequence(self):
        for i, structure in enumerate(self.output_structure):
            responses = self.run_call(structure, i)
            self.append_chat_history(responses)
        return responses

    def markdown_log(self, llm: Runnable):
        """Generate a markdown log file with a Mermaid chain graph."""
        with open(self.chain_graph_file, "w") as f:
            f.write(f"# {self.project_dir}\n\n{self.date}\n\n## chain graph\n\n")
            f.write(f"```mermaid\n{llm.get_graph().draw_mermaid()}\n```\n")

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
        self.output_structure = [output_structure]
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
        self.yaml_out = self.output_file.with_suffix(f".{self.run}.yml")
        self.chain_file = self.project_dir / Path("chain.py")
        self.chain_sequential = False
        self.chain_module = None

        # clear gpu memory
        torch.cuda.empty_cache()
        gc.collect()

    @timer(logger=logger)
    def execute(self):
        "Run the queries."
        # load prompts
        self.prompts = prompts_from_yaml(
            self.prompt_file.with_suffix(f".{self.run}.yml")
        )
        if self.sample:
            self.prompts = self.prompts[: self.sample]
        # define LLM
        self.llm: Runnable = self.chat_model(
            model=self.model,
            temperature=self.temperature,
            rate_limiter=self.rate_limiter().get(),
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            seed=self.seed,
        ).get()
        # get chain module
        if self.chain_file.exists():
            self.import_chain()
        # make calls
        if self.chain_sequential:
            logger.info("sequential calls")
            responses = self.run_chain_sequence()
            self.parse_output(responses, self.output_structure[-1])
            YamlLoader.save_yaml(
                self.prompts, self.output_dir / Path("chat_history.yml")
            )

        else:
            logger.info("parallel calls")
            for structure in self.output_structure:
                responses = self.run_call(structure)
                self.parse_output(responses, structure)

        # TODO this needs adapting w/ run_chain_sequence
        # if self.run == 1:
        #     self.markdown_log(llm)
        # self.save_yaml()
        # clear gpu memory
        # del llm
        # torch.cuda.empty_cache()
        # gc.collect()
        # self.markdown_log(llm)


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
        logger.info(f"run {run}")
        command.run = run
        command.execute()
