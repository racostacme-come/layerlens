"""Deterministic synthetic benchmarks; no external data are used."""

import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter
from numpy.polynomial.legendre import leggauss

from .solver import layer_profile, solve


def field_error(solution, exact, kind="linear", order=32):
    """L2 continuum error by per-edge Gauss-Legendre quadrature."""
    z, w = leggauss(order)
    h = np.diff(solution.x)
    points = solution.x[:-1, None] + h[:, None] * (z + 1) / 2
    error = solution.reconstruct(points, kind=kind) - exact(points)
    return float(np.sqrt(np.sum(h[:, None] * w * error**2 / 2)))


def run(output):
    """Write metrics, tables, two figures, and dependency provenance."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    pe = 100.0

    def exact(x):
        return layer_profile(x, pe)

    layer_rows, solutions = [], {}
    for n in (20, 40, 80, 160, 320):
        for scheme in ("central", "upwind", "sg"):
            s = solve(np.linspace(0, 1, n + 1), velocity=1, diffusivity=1 / pe, scheme=scheme)
            layer_rows.append(
                {
                    "cells": n,
                    "scheme": scheme,
                    "edge_peclet": pe / n,
                    "nodal_linf": float(np.max(np.abs(s.concentration - exact(s.x)))),
                    "linear_l2": field_error(s, exact),
                    "exponential_l2": field_error(s, exact, "exponential"),
                    "minimum": float(s.concentration.min()),
                    "maximum": float(s.concentration.max()),
                    "residual_linf": float(np.max(np.abs(s.residual))),
                    "balance_error": s.balance_error,
                }
            )
            if n == 20:
                solutions[scheme] = s
    manufactured_rows = []
    for n in (20, 40, 80, 160, 320):
        x = np.linspace(0, 1, n + 1)
        xi = x[1:-1]
        c = 1 + xi + np.sin(np.pi * xi)
        cp = 1 + np.pi * np.cos(np.pi * xi)
        cpp = -(np.pi**2) * np.sin(np.pi * xi)
        source = cp - (1 + xi) * cpp + 0.7 * c
        s = solve(
            x,
            velocity=2,
            diffusivity=1 + (x[:-1] + x[1:]) / 2,
            reaction=0.7,
            source=source,
            boundary=(1, 2),
        )

        def target(q):
            return 1 + q + np.sin(np.pi * q)

        manufactured_rows.append(
            {
                "cells": n,
                "nodal_linf": float(np.max(np.abs(s.concentration - target(x)))),
                "linear_l2": field_error(s, target),
                "exponential_l2": field_error(s, target, "exponential"),
                "residual_linf": float(np.max(np.abs(s.residual))),
                "balance_error": s.balance_error,
            }
        )
    fine, coarse = manufactured_rows[-1], manufactured_rows[-2]
    mms_order = float(np.log2(coarse["nodal_linf"] / fine["nodal_linf"]))
    # Independent quadrature refinement verifies the reported field norm.
    check = abs(
        field_error(solutions["sg"], exact, order=64)
        - field_error(solutions["sg"], exact, order=32)
    )
    metrics = {
        "schema_version": 1,
        "peclet": pe,
        "layer": layer_rows,
        "manufactured": manufactured_rows,
        "manufactured_final_order": mms_order,
        "quadrature_32_to_64_difference": check,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for name, rows in (("layer", layer_rows), ("manufactured", manufactured_rows)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    q = np.linspace(0.7, 1, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    colors = {"central": "#b84b40", "upwind": "#d19420", "sg": "#147d92"}
    for scheme, s in solutions.items():
        axes[0].plot(s.x, s.concentration, "o-", ms=3, label=scheme, color=colors[scheme])
    axes[0].plot(q, exact(q), "k--", lw=1.3, label="exact")
    axes[0].set(
        xlim=(0.7, 1),
        xlabel="Position x",
        ylabel="Concentration c",
        title="Coarse grid: Pe = 100, N = 20",
    )
    axes[0].legend(fontsize=8)
    s = solutions["sg"]
    axes[1].plot(q, exact(q), color="black", lw=2.5, label="exact")
    axes[1].plot(q, s.reconstruct(q), color=colors["sg"], label="SG + linear")
    axes[1].plot(
        q, s.reconstruct(q, kind="exponential"), "--", color="#7c4c9a", label="SG + exponential"
    )
    axes[1].scatter(s.x, s.concentration, s=16, color=colors["sg"], zorder=4)
    axes[1].set(
        xlim=(0.9, 1),
        xlabel="Position x",
        ylabel="Concentration c",
        title="Exact nodes, different fields",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(output / "boundary_layer.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for scheme in colors:
        rows = [row for row in layer_rows if row["scheme"] == scheme]
        axes[0].loglog(
            [row["cells"] for row in rows],
            [row["linear_l2"] for row in rows],
            "o-",
            label=scheme,
            color=colors[scheme],
        )
    axes[0].set(
        xlabel="Number of intervals N",
        ylabel="Linear reconstruction L2 error",
        title="Boundary layer: continuum error",
    )
    axes[0].legend(fontsize=8)
    for key, label in (
        ("nodal_linf", "nodal Linf"),
        ("linear_l2", "linear L2"),
        ("exponential_l2", "exponential L2"),
    ):
        axes[1].loglog(
            [row["cells"] for row in manufactured_rows],
            [row[key] for row in manufactured_rows],
            "o-",
            label=label,
        )
    ns = np.array([20, 320])
    axes[1].loglog(ns, 0.25 / ns**2, "k--", label="N^-2 reference")
    axes[1].set(
        xlabel="Number of intervals N",
        ylabel="Error",
        title="Variable diffusion + reaction + source",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks([20, 40, 80, 160, 320])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=0.2)
    fig.savefig(output / "convergence.png")
    plt.close(fig)
    provenance = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "matplotlib")},
    }
    (output / "environment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return metrics
