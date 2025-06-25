"""Module for the `prepare` command, to prepare LLM prompts."""

import gc
import os
from dataclasses import dataclass
from pathlib import Path

import click
import torch
import yaml
from langchain_core.prompts import ChatPromptTemplate

from lmf import embedding, prompt_template, vector_store
from lmf.command import Command
from lmf.io import YamlLoader, prompts_to_yaml

classes_dt = {
    "Embeddings": list(embedding.PARAMETER.types.keys()),
    "Vector_stores": list(vector_store.PARAMETER.types.keys()),
    "Prompt_templates": list(prompt_template.PARAMETER.types.keys()),
}
epilog = f"\b\nRECIPES\t*case insensitive*\n{yaml.dump(classes_dt)}"


@dataclass
class Prepare(Command):
    """Class to prepare prompts."""

    def __init__(
        self,
        ctx,
        k: int,
        embeddings: embedding.EmbeddingsType,
        model: str,
        vector_store: vector_store.VectorStoreType,
        prompt_template: prompt_template.PromptTemplateType,
        threshold: float,
    ):
        # input parameters
        self.ctx = ctx
        self.k = k
        self.embeddings = embeddings
        self.model = model
        self.vector_store = vector_store
        self.prompt_template = prompt_template
        self.threshold = threshold

        # explicit ctx
        self.base_dir: Path = ctx.obj["base_dir"]
        self.project_dir: Path = ctx.obj["project_dir"]
        self.prompt_dir: Path = ctx.obj["prompt_dir"]
        self.output_dir: Path = ctx.obj["output_dir"]
        self.log_dir: Path = ctx.obj["log_dir"]
        self.file_stem: Path = ctx.obj["file_stem"]
        self.runs: int = ctx.obj["runs"]
        self.date: str = ctx.obj["date"]

        # other parameters
        self.script = os.path.basename(__file__)
        self.run = 0
        self.times = {}
        self.warning = []
        self.prompt_file = self.prompt_dir / self.file_stem
        self.log_file = self.log_dir / self.prompt_dir.name / self.file_stem

        # load project files
        self.system = YamlLoader.load_yaml("system", self.project_dir)
        self.inputs = YamlLoader.load_yaml("inputs", self.project_dir)
        self.examples = YamlLoader.load_yaml("examples", self.project_dir)

        # clear gpu memory
        torch.cuda.empty_cache()
        gc.collect()

    def execute(self):
        "Run prompt preparation."

        self.now("start")
        # load embeddings
        dt = {}
        if self.run == self.runs and self.embeddings == embedding.Ollama:
            click.echo("... adding keep_alive = 0 to last Ollama request")
            dt = {"keep_alive": 0}

        _embeddings = self.embeddings(model=self.model, **dt).get()
        self.add_dumpd("embeddings-dump", _embeddings)
        _vector_store = self.vector_store(
            embeddings=_embeddings,
            examples=self.examples,
        ).get()
        prompt_template = self.prompt_template(
            vector_store=_vector_store,
            vectorstore_kwargs={"score_threshold": self.threshold},
            k=self.k,
        ).get()

        # define final_prompt
        messages = []
        if self.system:
            messages.append(("system", self.system))
        if prompt_template:
            messages.append(prompt_template)
        messages.append(("human", "{input}"))
        final_prompt = ChatPromptTemplate.from_messages(messages)
        self.add_dumpd("final_prompt-dump", final_prompt)

        # run example selector
        prompts = final_prompt.batch(self.inputs)

        # save prompts
        prompts_to_yaml(
            prompts=prompts, file=self.prompt_file.with_suffix(f".{self.run}.yml")
        )
        self.now("stop")
        self.save_yaml()

        # clear gpu memory
        del _embeddings
        del _vector_store
        del prompt_template
        del final_prompt
        torch.cuda.empty_cache()
        gc.collect()


@click.command(
    context_settings={"show_default": True},
    epilog=epilog,
)
@click.option(
    "-k",
    default=4,
    type=click.IntRange(min=0),
    help="Number of examples for few-shot prompting (k <= len(examples); 0 = all)",
)
@click.option(
    "-e",
    "--embeddings",
    default="HuggingFace",
    type=embedding.PARAMETER,
    help="A EmbeddingsType subclass from schema.py",
)
@click.option(
    "-m",
    "--model",
    default="Qwen/Qwen3-Embedding-0.6B",
    help="Name of embeddings model (download models beforehand)",
)
@click.option(
    "-v",
    "--vector-store",
    default="Memory",
    type=vector_store.PARAMETER,
    help="A vector_store.VectorStoreType subclass from schema.py",
)
@click.option(
    "-p",
    "--prompt-template",
    default="SemanticFewShotScore",
    type=prompt_template.PARAMETER,
    help="A prompt_template.PromptTemplateType subclass from prompt_template.py",
)
@click.option(
    "--threshold",
    default=0.0,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Minimum similarity to include examples (w/ `-p SemanticFewShot[Score]`)",
)
@click.pass_context
def prepare(
    ctx,
    k: int,
    embeddings: embedding.EmbeddingsType,
    model: str,
    vector_store: vector_store.VectorStoreType,
    prompt_template: prompt_template.PromptTemplateType,
    threshold: float,
):
    """Prepares LLM final prompts from a recipe of components."""

    command = Prepare(
        ctx=ctx,
        k=k,
        embeddings=embeddings,
        model=model,
        vector_store=vector_store,
        prompt_template=prompt_template,
        threshold=threshold,
    )
    # execute runs
    for run in range(1, ctx.obj["runs"] + 1):
        click.echo(f"... {command.script} - run - {run}")
        command.run = run
        command.execute()


if __name__ == "__main__":
    prepare()
