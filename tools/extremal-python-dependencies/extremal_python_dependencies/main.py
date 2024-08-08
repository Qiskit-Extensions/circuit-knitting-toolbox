# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import re
import configparser
from typing import List

import toml
import typer


# https://peps.python.org/pep-0508/#names
_name_re = re.compile(r"^([A-Z0-9][A-Z0-9._-]*[A-Z0-9])", re.IGNORECASE)


def mapfunc_replace(replacements: List[str]):
    """Use the provided version(s) of certain packages"""
    d: dict[str, str] = {}
    for r in replacements:
        match = _name_re.match(r)
        if match is None:
            raise RuntimeError(f"Replacement dependency does not match PEP 508: {r}")
        name = match.group()
        if name in d:
            raise RuntimeError("Duplicate name")
        d[name] = r

    def _mapfunc_replace(dep):
        # Replace https://peps.python.org/pep-0508/#names with provided
        # version, often a https://peps.python.org/pep-0440/#direct-references
        match = _name_re.match(dep)
        if match is None:
            raise RuntimeError(f"Dependency does not match PEP 508: `{dep}`")
        dep_name = match.group()
        return d.get(dep_name, dep)

    return _mapfunc_replace


def mapfunc_minimum(dep):
    """Set each dependency to its minimum version"""
    for clause in dep.split(","):
        if "*" in clause and "==" in clause:
            raise ValueError(
                "Asterisks in version specifiers are not currently supported "
                "by the minimum version tests.  We recommend using the "
                "'compatible release' operator instead: "
                "https://peps.python.org/pep-0440/#compatible-release"
            )
    return re.sub(r"[>~]=", r"==", dep)


def inplace_map(fun, lst: list):
    """In-place version of Python's `map` function"""
    for i, x in enumerate(lst):
        lst[i] = fun(x)


def process_dependencies_in_place(d: dict, mapfunc):
    """Given a parsed `pyproject.toml`, process dependencies according to `mapfunc`"""
    proj = d["project"]

    try:
        deps = proj["dependencies"]
    except KeyError:
        pass  # no dependencies; that's unusual, but fine.
    else:
        inplace_map(mapfunc, deps)

    try:
        opt_deps = proj["optional-dependencies"]
    except KeyError:
        pass  # no optional dependencies; that's fine.
    else:
        for dependencies_list in opt_deps.values():
            inplace_map(mapfunc, dependencies_list)

    try:
        build_system = d["build-system"]
    except KeyError:
        pass
    else:
        try:
            build_system_requires = build_system["requires"]
        except KeyError:
            pass
        else:
            inplace_map(mapfunc, build_system_requires)


app = typer.Typer()


@app.command()
def get_tox_minversion():
    """Extract tox minversion from `tox.ini`"""
    config = configparser.ConfigParser()
    config.read("tox.ini")
    print(config["tox"]["minversion"])


def _pin_dependencies(mapfunc, inplace: bool):
    with open("pyproject.toml") as f:
        d = toml.load(f)
    process_dependencies_in_place(d, mapfunc)

    # Modify pyproject.toml so hatchling will allow direct references
    # as dependencies.
    d.setdefault("tool", {}).setdefault("hatch", {}).setdefault("metadata", {})[
        "allow-direct-references"
    ] = True

    _save_pyproject_toml(d, inplace)


@app.command()
def pin_dependencies_to_minimum(inplace: bool = False):
    """Pin all dependencies in `pyproject.toml` to their minimum versions."""
    _pin_dependencies(mapfunc_minimum, inplace)


@app.command()
def pin_dependencies(replacements: List[str], inplace: bool = False):
    """Pin dependencies in `pyproject.toml` to the provided versions."""
    _pin_dependencies(mapfunc_replace(replacements), inplace)


@app.command()
def add_dependency(dependency: str, inplace: bool = False):
    """Add a dependency to `pyproject.toml`."""
    with open("pyproject.toml") as f:
        d = toml.load(f)
    d["project"]["dependencies"].append(dependency)
    _save_pyproject_toml(d, inplace)


def _save_pyproject_toml(d: dict, inplace: bool) -> None:
    if inplace:
        with open("pyproject.toml", "w") as f:
            toml.dump(d, f)
    else:
        print(toml.dumps(d))


def main():
    return app()
