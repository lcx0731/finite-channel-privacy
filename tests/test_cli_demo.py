import json
from fractions import Fraction as F
import subprocess
import sys

import pytest

from finite_privacy.demo import run_demo, write_outputs


def cli(path, *extra):
    return subprocess.run([sys.executable, "-m", "finite_privacy.cli", str(path), *extra],
                          capture_output=True, text=True)


def test_cli_exact_calibration_and_input_unchanged(tmp_path):
    path = tmp_path / "channel.json"
    original = '{"p":["3/4","1/4"],"q":["1/4","3/4"]}'
    path.write_text(original)
    result = cli(path, "--releases", "4", "--alpha", "9", "--target-delta", "0.01")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["compressed_states"] == 5
    assert F(report["minimum_alpha"]) == F(1961, 25)
    assert path.read_text() == original
    result = cli(path, "--releases", "0")
    assert json.loads(result.stdout)["minimum_alpha"] == "1"


@pytest.mark.parametrize("data", ['{}', '[]', '{"p":[0.5,0.5],"q":[1,0]}',
    '{"p":[true],"q":[1]}', '{"p":[NaN],"q":[1]}', '{"p":["1/0"],"q":[1]}',
    '{"p":[1],"p":[1],"q":[1]}', '{"p":[1],"q":[1],"typo":1}', '{'])
def test_cli_rejects_bad_json_and_inexact_probabilities(tmp_path, data):
    path = tmp_path / "bad.json"
    path.write_text(data)
    result = cli(path)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("args", [("--alpha", "0"), ("--alpha", "1/0"), ("--target-delta", "2"),
    ("--releases", "-1"), ("--releases", "101")])
def test_cli_rejects_bad_options(tmp_path, args):
    path = tmp_path / "channel.json"
    path.write_text('{"p":[1],"q":[1]}')
    result = cli(path, *args)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_demo_reproducibility_and_reference_results(tmp_path):
    summary = write_outputs(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text()) == run_demo()
    assert (tmp_path / "privacy-profiles.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert [r["ratio_states"] for r in summary["calibration"]] == [2, 5, 13]
    assert summary["postprocessing"]["majority_pure_alpha"] == "27/5"
    assert summary["shared_coin_counterexample"]["actual_joint_delta_floor"] == "1"


def test_demo_entrypoint(tmp_path):
    output = tmp_path / "demo"
    result = subprocess.run([sys.executable, "-m", "finite_privacy.demo", "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output / "summary.json").exists()
    assert json.loads(result.stdout)["calibration"][-1]["transcript_checks_passed"]
