from pathlib import Path

from click.testing import CliRunner

from clilm.main import cli


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


def test_prepare():
    runner = CliRunner()

    ls = [
        "--clear",
        "-p",
        "wizard-of-math",
        "prepare",
    ]
    result = runner.invoke(cli, ls)
    assert result.exit_code == 0
    assert Path("projects/wizard-of-math/prompt/run.1.yml").exists()


def test_query():
    runner = CliRunner()
    ls = [
        "--clear",
        "-p",
        "wizard-of-math",
        "prepare",
        "query",
    ]
    result = runner.invoke(cli, ls)
    assert result.exit_code == 0
    assert Path("projects/wizard-of-math/output/run.1.yml").exists()
    assert Path("projects/wizard-of-math/prompt/run.1.yml").exists()


def test_empty():
    runner = CliRunner()
    ls = [
        "-p",
        "wizard-of-math",
        "prepare",
        "query",
    ]
    result = runner.invoke(cli, ls)
    result = runner.invoke(cli, ["empty", "wizard-of-math"])
    assert result.exit_code == 0
    assert not Path("projects/wizard-of-math/log").exists()
    assert not Path("projects/wizard-of-math/output").exists()
    assert not Path("projects/wizard-of-math/prompt").exists()
