#!/usr/bin/env python3
"""Install ABot-Recon on a Jetson (aarch64) without clobbering JetPack PyTorch.

The documented install (`pip install torch==2.5.1 --index-url .../whl/cu121`)
is x86-only, and plain `pip install -e .` is worse than useless on a Jetson:
PyPI ships a CPU-only `linux_aarch64` torch wheel, so resolving the `torch==`
pin silently replaces a working CUDA build with one that cannot see the GPU.

So this script never resolves torch itself. It requires a CUDA-capable torch
to already be importable (JetPack wheel, NVIDIA pip index, or a container),
verifies it, then installs the remaining dependencies under a constraints file
pinning that exact torch build — local version segment included — so any
transitive torch requirement (pypose, for instance) resolves to what is already
installed. The package itself goes in with `--no-deps`, since its `torch==2.5.1`
pin can never be satisfied by a JetPack build.

    python scripts/setup_jetson_env.py --dry-run     # show the plan
    python scripts/setup_jetson_env.py               # run it

Inference on the Orin must use the pure-PyTorch attention path — FlashInfer
and a compiled cuRoPE are neither installed nor needed:

    python demo.py --image-dir <frames> --output-dir outputs/g1 \\
      --attention-backend sdpa --no-loop-closure --save-world-points \\
      --stride 2 --dense-stride 4

Expect throughput far below the 24 FPS H100 benchmark; the ~6.71 GiB working
set does fit Orin unified memory.
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Dependencies whose version pins exist for the x86 CUDA release and must be
# left to JetPack on aarch64.
TORCH_PACKAGES = ("torch", "torchvision")
DEFAULT_EXTRAS = ("loop",)


class SetupError(RuntimeError):
    """A precondition this script refuses to work around."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter whose environment is installed into",
    )
    parser.add_argument(
        "--extras",
        default="",
        help=f"Comma-separated optional extras to install ({', '.join(DEFAULT_EXTRAS)})",
    )
    parser.add_argument(
        "--allow-non-aarch64",
        action="store_true",
        help="Skip the aarch64 check (for rehearsing the plan on a workstation)",
    )
    parser.add_argument(
        "--allow-cpu-torch",
        action="store_true",
        help="Proceed even when the installed torch has no CUDA support",
    )
    return parser


def parse_requirement_name(requirement: str) -> str:
    """Distribution name of a PEP 508 requirement string."""
    match = re.match(r"^\s*([A-Za-z0-9._-]+)", requirement)
    if not match:
        raise ValueError(f"unparseable requirement: {requirement!r}")
    return match.group(1).replace("_", "-").lower()


def read_dependencies(pyproject: Path) -> list[str]:
    """Runtime dependencies declared in pyproject.toml, in declaration order."""
    import tomllib

    data = tomllib.loads(pyproject.read_text())
    return list(data["project"]["dependencies"])


def read_extra_dependencies(pyproject: Path, extras: list[str]) -> list[str]:
    import tomllib

    data = tomllib.loads(pyproject.read_text())
    optional = data["project"].get("optional-dependencies", {})
    collected: list[str] = []
    for extra in extras:
        if extra not in optional:
            raise SetupError(f"unknown extra {extra!r}; available: {', '.join(sorted(optional))}")
        collected.extend(optional[extra])
    return collected


def jetson_requirements(dependencies: list[str]) -> list[str]:
    """Drop the torch pins so the JetPack build stays installed."""
    return [
        requirement
        for requirement in dependencies
        if parse_requirement_name(requirement) not in TORCH_PACKAGES
    ]


def torch_constraints(report: dict[str, str]) -> str:
    """Constraints text pinning the installed torch build.

    Pinning the full version — including the `+nv...` local segment — makes pip
    treat the requirement as already satisfied instead of downloading the
    CPU-only PyPI wheel over it.
    """
    lines = [f"torch=={report['version']}"]
    torchvision = report.get("torchvision")
    if torchvision:
        lines.append(f"torchvision=={torchvision}")
    return "\n".join(lines) + "\n"


def build_commands(
    *,
    python: str,
    requirements: list[str],
    repo_root: Path = REPO_ROOT,
    constraints: Path | None = None,
) -> list[list[str]]:
    """The pip invocations, in order."""
    pip = [python, "-m", "pip", "install"]
    constrained = [*pip, "-c", str(constraints)] if constraints else list(pip)
    commands = []
    if requirements:
        commands.append([*constrained, *requirements])
    # The editable install alone skips dependency resolution: pyproject pins
    # torch==2.5.1, which no JetPack build satisfies.
    commands.append([*pip, "--no-deps", "-e", str(repo_root)])
    return commands


def detect_platform(machine: str | None = None) -> str:
    return machine or platform.machine()


def is_jetson(release_file: Path = Path("/etc/nv_tegra_release")) -> bool:
    """Whether this is a Tegra board (Orin, Xavier, ...)."""
    return release_file.exists()


def torch_cuda_report(python: str) -> dict[str, str]:
    """Version/CUDA/device facts from the target interpreter's torch."""
    probe = (
        "import json, torch;"
        "\ntry:\n import torchvision; tv = torchvision.__version__\n"
        "except Exception:\n tv = ''\n"
        "print(json.dumps({'version': torch.__version__,"
        " 'torchvision': tv,"
        " 'cuda': str(torch.version.cuda),"
        " 'available': str(torch.cuda.is_available()),"
        " 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}))"
    )
    result = subprocess.run(
        [python, "-c", probe], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SetupError(
            "torch is not importable in the target environment. Install a CUDA-capable "
            "aarch64 build first (JetPack wheel or NVIDIA pip index), then re-run.\n"
            f"{result.stderr.strip()}"
        )
    import json

    return json.loads(result.stdout.strip().splitlines()[-1])


def check_torch(report: dict[str, str], *, allow_cpu: bool) -> None:
    if report.get("cuda") in (None, "", "None") or report.get("available") != "True":
        message = (
            f"torch {report.get('version')} reports CUDA={report.get('cuda')} "
            f"available={report.get('available')} — this looks like the PyPI CPU-only "
            "aarch64 wheel. Installing over it will not give you GPU inference."
        )
        if not allow_cpu:
            raise SetupError(message + " Pass --allow-cpu-torch to proceed anyway.")
        print(f"WARNING: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    machine = detect_platform()
    if machine != "aarch64" and not args.allow_non_aarch64:
        print(
            f"This script targets aarch64 Jetson boards; detected {machine}. "
            "On x86 follow the README install. Use --allow-non-aarch64 to rehearse.",
            file=sys.stderr,
        )
        return 2

    extras = [extra for extra in args.extras.split(",") if extra]
    pyproject = REPO_ROOT / "pyproject.toml"
    try:
        requirements = jetson_requirements(read_dependencies(pyproject))
        requirements.extend(jetson_requirements(read_extra_dependencies(pyproject, extras)))
        constraints = None
        if args.dry_run:
            commands = build_commands(python=args.python, requirements=requirements)
        else:
            report = torch_cuda_report(args.python)
            print(
                f"torch {report['version']} (CUDA {report['cuda']}, "
                f"available={report['available']}, device={report['device'] or 'none'})"
            )
            check_torch(report, allow_cpu=args.allow_cpu_torch)
            constraints = REPO_ROOT / "build" / "jetson-torch-constraints.txt"
            constraints.parent.mkdir(parents=True, exist_ok=True)
            constraints.write_text(torch_constraints(report))
            print(f"constraints: {constraints} -> {torch_constraints(report).strip()!r}")
            commands = build_commands(
                python=args.python, requirements=requirements, constraints=constraints
            )
    except SetupError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if not is_jetson() and not args.allow_non_aarch64:
        print("warning: /etc/nv_tegra_release is absent; is this really a Jetson?", file=sys.stderr)

    for command in commands:
        print("$ " + " ".join(command))
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"error: command failed with {result.returncode}", file=sys.stderr)
            return result.returncode
    if args.dry_run:
        print("\n(dry run — nothing installed)")
    else:
        print("\nInstalled. Run inference with --attention-backend sdpa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
