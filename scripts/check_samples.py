"""Reproduce metrics in a separate directory and compare at declared tolerances."""

import json
from pathlib import Path

import numpy as np

from layerlens.experiments import run

ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual, path="metrics"):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            compare(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), path
        for i, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(left, right, f"{path}/{i}")
    elif isinstance(expected, float):
        # Algebraic defects are roundoff-sensitive; discretization errors use
        # a tight relative check. Neither comparison tests image byte identity.
        atol = 2e-10 if "residual" in path or "balance" in path else 2e-12
        np.testing.assert_allclose(actual, expected, rtol=5e-7, atol=atol, err_msg=path)
    else:
        assert expected == actual, path


if __name__ == "__main__":
    expected = json.loads((ROOT / "examples/results/metrics.json").read_text())
    actual = run(ROOT / "output/reproduction")
    compare(expected, actual)
    assert 1.98 < actual["manufactured_final_order"] < 2.02
    assert actual["quadrature_32_to_64_difference"] < 1e-12
    print("Recorded metrics reproduced; convergence and quadrature checks passed.")
