"""Small command-line entry points for reproducible experiments."""

import argparse
from pathlib import Path

import numpy as np

from .solver import solve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="reproduce the manuscript experiments")
    demo.add_argument("--output", type=Path, default=Path("output"))
    single = sub.add_parser("solve", help="solve the normalized constant-coefficient problem")
    single.add_argument("--cells", type=int, default=20)
    single.add_argument("--peclet", type=float, default=100)
    single.add_argument("--scheme", choices=("sg", "central", "upwind"), default="sg")
    single.add_argument("--output", type=Path, default=Path("output/profile.csv"))
    args = parser.parse_args(argv)
    if args.command == "demo":
        from .experiments import run

        metrics = run(args.output)
        print(
            f"Wrote experiments to {args.output}; MMS order = "
            f"{metrics['manufactured_final_order']:.6f}"
        )
    else:
        if args.cells < 2:
            parser.error("--cells must be at least 2")
        if not np.isfinite(args.peclet):
            parser.error("--peclet must be finite")
        result = solve(
            np.linspace(0, 1, args.cells + 1),
            velocity=args.peclet,
            diffusivity=1,
            scheme=args.scheme,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.output,
            np.column_stack((result.x, result.concentration)),
            delimiter=",",
            header="x,concentration",
            comments="",
        )
        print(f"Wrote {args.output}; max flux residual = {max(abs(result.residual)):.3e}")


if __name__ == "__main__":
    main()
