from pathlib import Path

from click.testing import CliRunner

from lmf.main import cli

base_dir = ["--base-dir", "tests/projects"]
project = ["-p", "wizard-of-math"]

log_dir = Path(f"{base_dir[1]}/{project[1]}/log")
output_dir = Path(f"{base_dir[1]}/{project[1]}/output")
prompt_dir = Path(f"{base_dir[1]}/{project[1]}/prompt")


def test_chain():
    runner = CliRunner()
    cmd = (
        base_dir
        + project
        + [
            "clear",
            "prepare",
            "query",
        ]
    )
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0
