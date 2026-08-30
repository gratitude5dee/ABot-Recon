import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "setup_jetson_env.py"


def load_module():
    spec = importlib.util.spec_from_file_location("setup_jetson_env", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_module()


def test_torch_pins_are_dropped_but_everything_else_survives():
    dependencies = setup.read_dependencies(REPO_ROOT / "pyproject.toml")
    names = {setup.parse_requirement_name(r) for r in setup.jetson_requirements(dependencies)}
    assert "torch" not in names and "torchvision" not in names
    assert {"einops", "numpy", "safetensors", "pillow"} <= names


def test_dependencies_install_under_the_torch_constraint_and_the_package_without_deps(tmp_path):
    constraints = tmp_path / "c.txt"
    commands = setup.build_commands(
        python="/usr/bin/python3",
        requirements=["numpy>=1.24"],
        repo_root=REPO_ROOT,
        constraints=constraints,
    )
    assert commands[0][-3:] == ["-c", str(constraints), "numpy>=1.24"]
    assert "--no-deps" not in commands[0]
    assert commands[-1][-3:] == ["--no-deps", "-e", str(REPO_ROOT)]
    assert "-c" not in commands[-1]


def test_constraints_pin_the_local_jetpack_version_exactly():
    text = setup.torch_constraints(
        {"version": "2.5.0a0+872d972e41.nv24.08", "torchvision": "0.20.0a0+afc54f7"}
    )
    assert text.splitlines() == [
        "torch==2.5.0a0+872d972e41.nv24.08",
        "torchvision==0.20.0a0+afc54f7",
    ]
    assert setup.torch_constraints({"version": "2.5.1"}).splitlines() == ["torch==2.5.1"]


def test_no_requirements_still_installs_the_package():
    commands = setup.build_commands(python="python", requirements=[], repo_root=REPO_ROOT)
    assert len(commands) == 1


def test_cpu_only_torch_is_refused_unless_explicitly_allowed():
    cpu = {"version": "2.5.1", "cuda": "None", "available": "False", "device": ""}
    # This is the exact shape of the trap: the PyPI aarch64 wheel installs
    # cleanly and reports no CUDA.
    with pytest.raises(setup.SetupError):
        setup.check_torch(cpu, allow_cpu=False)
    setup.check_torch(cpu, allow_cpu=True)
    setup.check_torch(
        {"version": "2.5.0", "cuda": "12.6", "available": "True", "device": "Orin"},
        allow_cpu=False,
    )


def test_unknown_extras_are_named():
    with pytest.raises(setup.SetupError) as error:
        setup.read_extra_dependencies(REPO_ROOT / "pyproject.toml", ["nope"])
    assert "loop" in str(error.value)


def test_loop_extra_resolves():
    extra = setup.read_extra_dependencies(REPO_ROOT / "pyproject.toml", ["loop"])
    assert {setup.parse_requirement_name(r) for r in extra} >= {"faiss-cpu", "scipy"}


def test_dry_run_prints_the_plan_without_touching_pip(capsys, monkeypatch):
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: pytest.fail("pip ran"))
    assert setup.main(["--dry-run", "--allow-non-aarch64"]) == 0
    out = capsys.readouterr().out
    assert "--no-deps" in out and "dry run" in out


def test_non_aarch64_without_the_override_is_a_hard_stop(monkeypatch):
    monkeypatch.setattr(setup, "detect_platform", lambda machine=None: "x86_64")
    assert setup.main([]) == 2
