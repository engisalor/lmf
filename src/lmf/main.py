"""CLI entrypoint."""

import random
import shutil
from datetime import datetime
from pathlib import Path

import click
from langchain.globals import set_debug
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from lmf.prepare import prepare
from lmf.query import query


@click.group(
    chain=True,
    context_settings={"show_default": True},
)
@click.option(
    "-p",
    "--project",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help="Current project (generally a subdirectory within `./projects`)",
)
@click.option(
    "-r",
    "--runs",
    default=1,
    type=click.IntRange(min=1),
    help="Number of runs to repeat",
)
@click.option(
    "--base-dir",
    default="projects",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help="Base directory containing projects",
)
@click.option(
    "-f",
    "--file-stem",
    default="run",
    help="Base name used for generated file",
)
@click.option("--debug/--no-debug", default=False, help="Set langchain global debug")
@click.option(
    "--cache/--no-cache",
    default=False,
    help="Use local LLM cache (existing cache is deleted if False)",
)
@click.option(
    "--seed",
    default=0,
    type=click.IntRange(min=0),
    help="Randomization seed (0 == None; a positive integer sets the seed)",
)
@click.pass_context
def cli(
    ctx,
    project: str,
    runs: int,
    base_dir: str | Path,
    file_stem: str | Path,
    debug: bool,
    cache: bool,
    seed: int,
):
    """A CLI for running language models with langchain."""
    if project:
        # debug
        set_debug(debug)

        # sqlite caching
        if cache:
            set_llm_cache(SQLiteCache(database_path=".langchain.db"))
        else:
            Path(".langchain.db").unlink(missing_ok=True)
            set_llm_cache(None)

        # parameters
        project_dir = Path(base_dir) / Path(project)
        log_dir = project_dir / Path("log")
        prompt_dir = project_dir / Path("prompt")
        output_dir = project_dir / Path("output")
        if not seed:
            seed = None
        random.seed(seed)
        print(f"... seed = {seed}")

        # ctx
        ctx.ensure_object(dict)
        ctx.obj["base_dir"] = Path(base_dir)
        ctx.obj["project_dir"] = project_dir
        ctx.obj["prompt_dir"] = prompt_dir
        ctx.obj["output_dir"] = output_dir
        ctx.obj["log_dir"] = log_dir
        ctx.obj["runs"] = runs
        ctx.obj["date"] = datetime.now().isoformat(timespec="seconds")
        ctx.obj["file_stem"] = Path(file_stem)
        ctx.obj["seed"] = seed

        # check project path
        if not project_dir.exists():
            m = f"No such directory: {project_dir}"
            raise FileNotFoundError(m)

        # make directories
        for dir in [
            prompt_dir,
            output_dir,
            log_dir,
            (log_dir / prompt_dir.name),
            (log_dir / output_dir.name),
        ]:
            dir.mkdir(exist_ok=True)


@cli.command(context_settings={"show_default": True})
@click.pass_context
def clear(ctx):
    """Deletes a project's generated subdirectories (prompt/, output/, log/)."""

    project_dir = ctx.obj["project_dir"]
    log_dir = ctx.obj["log_dir"]
    prompt_dir = ctx.obj["prompt_dir"]
    output_dir = ctx.obj["output_dir"]

    if not project_dir.exists():
        m = f"No such directory: {project_dir}"
        raise FileNotFoundError(m)

    # clear directories
    click.echo("... clearing generated subdirectories")
    for dir in [
        prompt_dir,
        output_dir,
        log_dir,
        (log_dir / prompt_dir.name),
        (log_dir / output_dir.name),
    ]:
        shutil.rmtree(dir, ignore_errors=True)
        dir.mkdir(exist_ok=True)


cli.add_command(query)
cli.add_command(prepare)

if __name__ == "__main__":
    cli(obj={})
