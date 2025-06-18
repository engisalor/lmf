from pathlib import Path

from click.testing import CliRunner

from clilm.main import cli

base_dir = ["--base-dir", "tests/projects"]
project = ["-p", "wizard-of-math"]

log_dir = Path(f"{base_dir[1]}/{project[1]}/log")
output_dir = Path(f"{base_dir[1]}/{project[1]}/output")
prompt_dir = Path(f"{base_dir[1]}/{project[1]}/prompt")


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.stdout.startswith("Usage:")


def test_prepare_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["prepare", "--help"])
    assert result.stdout.startswith("Usage:")


def test_query_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["query", "--help"])
    assert result.stdout.startswith("Usage:")


def test_clear_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["clear", "--help"])
    assert result.stdout.startswith("Usage:")


def test_clear():
    runner = CliRunner()
    for dir in [log_dir, output_dir, prompt_dir]:
        dir.mkdir(exist_ok=True)
        (dir / Path("subdirectory")).mkdir(exist_ok=True)
        assert (dir / Path("subdirectory")).exists()
    result = runner.invoke(cli, base_dir + project + ["clear"])
    for dir in [log_dir, output_dir, prompt_dir]:
        assert not (dir / Path("subdirectory")).exists()
