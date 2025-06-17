"""CLI entrypoint."""

import shutil
from datetime import datetime
from pathlib import Path

import click
from langchain.globals import set_debug
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from clilm.prepare import prepare
from clilm.query import query


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
    "--clear/--no-clear", default=False, help="Clear existing project subdirectories"
)
@click.option(
    "--base_dir",
    default="projects",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help="Base directory containing projects",
)
@click.option(
    "-f",
    "--file_stem",
    default="run",
    help="Base name used for generated file",
)
@click.option("--debug/--no-debug", default=False, help="Set langchain global debug")
@click.option(
    "--cache/--no-cache",
    default=False,
    help="Use local LLM cache (existing cache is deleted if False)",
)
@click.pass_context
def cli(
    ctx,
    project: str,
    runs: int,
    clear: bool,
    base_dir: str | Path,
    file_stem: str | Path,
    debug: bool,
    cache: bool,
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

        # ctx
        ctx.ensure_object(dict)
        ctx.obj["base_dir"] = Path(base_dir)
        ctx.obj["project_dir"] = project_dir
        ctx.obj["prompt_dir"] = prompt_dir
        ctx.obj["output_dir"] = output_dir
        ctx.obj["log_dir"] = log_dir
        ctx.obj["runs"] = runs
        ctx.obj["clear"] = clear
        ctx.obj["date"] = datetime.now().isoformat(timespec="seconds")
        ctx.obj["file_stem"] = Path(file_stem)

        # check project path
        if not project_dir.exists():
            m = f"No such directory: {project_dir}"
            raise FileNotFoundError(m)

        if clear:
            click.echo(f"... clearing {log_dir}")
            shutil.rmtree(log_dir, ignore_errors=True)
        log_dir.mkdir(exist_ok=True)
        (log_dir / prompt_dir.name).mkdir(exist_ok=True)
        (log_dir / output_dir.name).mkdir(exist_ok=True)


@cli.command(context_settings={"show_default": True})
@click.argument("project")
@click.option(
    "--base_dir",
    default="projects",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help="Base directory containing projects",
)
def empty(
    project: str | Path,
    base_dir: str | Path,
):
    """Deletes a project's generated directories (prompt, output, log)."""
    project_dir = Path(base_dir) / Path(project)
    log_dir = project_dir / Path("log")
    prompt_dir = project_dir / Path("prompt")
    output_dir = project_dir / Path("output")
    if not project_dir.exists():
        m = f"No such directory: {project_dir}"
        raise FileNotFoundError(m)

    click.echo(f"... clearing {project_dir}")
    shutil.rmtree(log_dir, ignore_errors=True)
    shutil.rmtree(prompt_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)


cli.add_command(query)
cli.add_command(prepare)

if __name__ == "__main__":
    cli(obj={})
