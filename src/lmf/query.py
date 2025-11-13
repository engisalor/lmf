"""Module for the `query` command, to make LLM queries."""

import gc
import importlib.util
import json
import os
import sys
from ast import literal_eval
from datetime import datetime
from pathlib import Path

import click
import torch
import yaml
from langchain.chat_models.base import BaseChatModel
from langchain_core.callbacks import get_usage_metadata_callback
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

    def append_prompt(self, msg, structure, prompts):
        for prompt in prompts:
            prompt.messages[-1].content = prompt.messages[-1].content.replace(msg, "")
        if structure.__name__.startswith("Unstructured"):
            w = f"{structure.__name__} - temporarily appending markdown instructions"
            logger.warning(w)
            for prompt in prompts:
                prompt.messages[-1].content += msg
        return prompts

    @timer(logger=logger)
    def run_call(self, structure, i=0):
        # prompt sampling and randomization logic
        if self.sample and not self.random:
            logger.info(f"ordered sampling - {self.sample}/{len(self.prompts)}")
            prompts = self.prompts[: self.sample].copy()
        # TODO implement randomized sampling (requires bypass of prepare cmd)
        # elif self.sample and self.random:
        #     logger.info(f"randomized sampling - {self.sample}/{len(self.prompts)}")
        #     prompts = random.sample(self.prompts.copy(), self.sample)
        else:
            prompts = self.prompts.copy()
        # thinking logic
        # if structure.__name__.endswith("Think"):
        #     think = True
        # else:
        #     think = self.think
        # create model
        params = self.chat_model_param.copy()
        params |= dict(
            model=self.model,
            temperature=self.temperature,
            rate_limiter=self.rate_limiter().get(),
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            seed=self.seed,
            # think=think,
        )
        logger.debug(f"chat model params {params}")
        llm: BaseChatModel = self.chat_model(**params).get()
        if not structure.__name__.startswith("Unstructured"):
            llm = llm.with_structured_output(structure)
        # append msg to get better organized markdown for Unstructured
        msg = "\n**NOTE**\n\nOutput the categories in markdown format, with the category label as a heading (starting with ##) and with a new line starting with a dash for each item"
        prompts = self.append_prompt(msg, structure, prompts)
        # run batch
        u_file = Path(".lmf-usage.log")
        with get_usage_metadata_callback() as cb:
            responses = llm.batch(prompts)
            dt_usage = cb.usage_metadata
            dt_usage["date"] = datetime.now().isoformat()
            with open(u_file, "a") as f:
                f.write(json.dumps(dt_usage, sort_keys=True) + "\n")

        self.yaml_out = self.output_file.with_suffix(
            f".{self.model.replace('.', '~')}.{structure.__name__}.{self.run}.yml"
        )
        logger.info(f"{i} - '{self.yaml_out}'")
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
        chat_model_param: tuple,
        output_structure: BaseModel,
        sample: int,
        random: bool,
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
        self.random = random
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.think = think
        self.rate_limiter = rate_limiter
        chat_model_param = [x.strip().split("=") for x in chat_model_param]
        chat_model_param = {x[0].strip(): literal_eval(x[1]) for x in chat_model_param}
        self.chat_model_param: dict = chat_model_param

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
        self.chain_file = self.project_dir / Path("chain.py")
        self.chain_sequential = False
        self.chain_module = None

        # clear gpu memory
        torch.cuda.empty_cache()
        gc.collect()

    @timer(logger=logger)
    def execute(self):
        """Run the queries."""
        self.yaml_out = self.output_file.with_suffix(f".{self.run}.yml")
        # load prompts
        self.prompts = prompts_from_yaml(
            self.prompt_file.with_suffix(f".{self.run}.yml")
        )
        # get chain module
        if self.chain_file.exists():
            self.import_chain()
        # make calls
        if self.chain_sequential:
            logger.info("sequential calls")
            responses = self.run_chain_sequence()
            self.parse_output(responses, self.output_structure[-1])
            YamlLoader.save_yaml(
                self.prompts, self.output_dir / Path(f"chat-history.{self.run}.yml")
            )
        else:
            logger.info("parallel calls")
            for structure in self.output_structure:
                responses = self.run_call(structure)
                self.parse_output(responses, structure)

        # TODO this needs rewriting for use w/ sequential/parallel calls
        # self.markdown_log(llm)
        torch.cuda.empty_cache()
        gc.collect()
        self.save_yaml()


@click.command(context_settings={"show_default": True}, epilog=epilog)
@click.option(
    "-m",
    "--model",
    default="qwen3:1.7b",
    help="Name of model (download models beforehand)",
)
@click.option(
    "--chat-model",
    default="Ollama",
    type=chat_model.PARAMETER,
    help="A chat model chat model class from chat.py",
)
@click.option(
    "--chat-model-param",
    multiple=True,
    help="A parameter to pass to the chat model in the format 'key=value'",
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
@click.option("--random/--no-random", default=False, help="Toggle sample randomization")
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
    "--max-tokens",
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
    chat_model_param: tuple[str],
    output_structure: ModelMetaclass,
    sample: int,
    random: bool,
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
        chat_model_param=chat_model_param,
        output_structure=output_structure,
        sample=sample,
        random=random,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        think=think,
        rate_limiter=rate_limiter,
    )
    # execute runs
    for run in range(1, ctx.obj["runs"] + 1):
        logger.info(f"run {run}")
        setattr(command, "run", run)
        command.execute()
