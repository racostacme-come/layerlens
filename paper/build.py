"""Build the data-linked LaTeX paper and enforce a five-page maximum."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", help="path/name of tectonic or pdflatex")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    root = folder.parent
    data = json.loads((root / "examples/results/metrics.json").read_text())
    coarse = {r["scheme"]: r for r in data["layer"] if r["cells"] == 20}
    macros = []
    for name, value in {
        "NodeError": coarse["sg"]["nodal_linf"],
        "LinearError": coarse["sg"]["linear_l2"],
        "FittedError": coarse["sg"]["exponential_l2"],
        "QuadratureError": data["quadrature_32_to_64_difference"],
        "FineMms": data["manufactured"][-1]["nodal_linf"],
        "MaxResidual": max(r["residual_linf"] for r in data["manufactured"]),
    }.items():
        mantissa, exponent = f"{value:.3e}".split("e")
        macros.append(rf"\newcommand{{\{name}}}{{{mantissa}\times10^{{{int(exponent)}}}}}")
    macros.append(r"\newcommand{\MmsOrder}{" + f"{data['manufactured_final_order']:.5f}" + "}")
    rows = []
    for name, r in coarse.items():
        rows.append(
            f"{name.upper()} & {r['minimum']:.5f} & "
            f"{r['nodal_linf']:.3e} & {r['linear_l2']:.5f} " + r"\\"
        )
    macros.append(r"\newcommand{\LayerRows}{" + "\n".join(rows) + "}")
    rows = []
    for r in data["manufactured"]:
        rows.append(
            f"{r['cells']} & {r['nodal_linf']:.3e} & {r['linear_l2']:.3e} & "
            f"{r['exponential_l2']:.3e} " + r"\\"
        )
    macros.append(r"\newcommand{\MmsRows}{" + "\n".join(rows) + "}")
    (folder / "numbers.tex").write_text("\n".join(macros) + "\n", newline="\n")
    engine = args.engine or shutil.which("tectonic") or shutil.which("pdflatex")
    if not engine:
        raise SystemExit("Install Tectonic or TeX Live (pdflatex), then retry.")
    if Path(engine).exists():
        engine = str(Path(engine).resolve())
    name = Path(engine).stem.lower()
    build = folder / "build"
    build.mkdir(exist_ok=True)
    if name == "tectonic":
        command = [
            engine,
            "--keep-logs",
            "--keep-intermediates",
            "--outdir",
            str(build),
            "paper.tex",
        ]
        passes = 1
    elif name == "pdflatex":
        command = [
            engine,
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(build),
            "paper.tex",
        ]
        passes = 2
    else:
        raise SystemExit("Supported engines: tectonic or pdflatex")
    for _ in range(passes):
        result = subprocess.run(
            command, cwd=folder, capture_output=True, text=True, errors="replace", timeout=180
        )
        if result.returncode:
            raise SystemExit(result.stdout + result.stderr)
    log = (build / "paper.log").read_text(errors="replace")
    failures = re.findall(
        r"Overfull \\[hv]box[^\n]*|[^\n]*(?:undefined references|"
        r"Citation .* undefined|Reference .* undefined)[^\n]*",
        log,
    )
    if failures:
        raise SystemExit("Layout/reference errors:\n" + "\n".join(failures))
    pdf = build / "paper.pdf"
    reader = PdfReader(pdf)
    if not 1 <= len(reader.pages) <= 5:
        raise SystemExit(f"Invalid page count: {len(reader.pages)} (limit 5)")
    if any(not (p.extract_text() or "").strip() for p in reader.pages):
        raise SystemExit("Empty manuscript page")
    inputs = [folder / "paper.tex", folder / "numbers.tex", folder / "build.py"]
    inputs += sorted((root / "src").rglob("*.py"))
    inputs += sorted((root / "examples/results").glob("*"))
    report = {
        "schema_version": 1,
        "pages": len(reader.pages),
        "page_limit": 5,
        "engine": name,
        "overfull_boxes": 0,
        "undefined_references": 0,
        "inputs_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs
        },
    }
    shutil.copyfile(pdf, folder / "paper.pdf")
    (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n", newline="\n")
    print(f"Compiled {len(reader.pages)} pages; no overfull boxes or undefined references.")


if __name__ == "__main__":
    main()
