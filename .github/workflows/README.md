# GitHub Actions workflows

This directory contains a number of workflows for use with [GitHub Actions](https://docs.github.com/actions).  They specify what standards should be expected for development of this software, including pull requests.  These workflows are designed to work out of the box for any research software prototype, especially those based on [Qiskit](https://qiskit.org/).

## Lint check (`lint.yml`)

This workflow checks that the code is formatted properly and follows the style guide by installing tox and running the [lint environment](/tests/#lint-environment) (`tox -e lint`).

## Latest version tests (`test_latest_versions.yml`)

This workflow installs the latest version of tox and runs [the current repository's tests](/tests/#test-py-environments) under each supported Python version on Linux and under a single Python version on macOS and Windows.  This is the primary testing workflow.  It runs for all code changes and additionally once per day, to ensure tests continue to pass as new versions of dependencies are released.

## Development version tests (`test_development_versions.yml`)

This workflow installs tox and modifies `pyproject.toml` to use the _development_ versions of certain Qiskit packages, using [extremal-python-dependencies](https://github.com/IBM/extremal-python-dependencies).  For all other packages, the latest version is installed.  This workflow runs on two versions of Python: the minimum supported version and the maximum supported version.  Its purpose is to identify as soon as possible (i.e., before a Qiskit release) when changes in Qiskit will break the current repository.  This workflow runs for all code changes, as well as on a timer once per day.

## Minimum version tests (`test_minimum_versions.yml`)

This workflow first installs the minimum supported tox version (the `minversion` specified in [`tox.ini`](/tox.ini)) and then installs the _minimum_ compatible version of each package listed in `pyproject.toml`, using [extremal-python-dependencies](https://github.com/IBM/extremal-python-dependencies).  The purpose of this workflow is to make sure the minimum version specifiers in these files are accurate, i.e., that the tests actually pass with these versions.  This workflow uses a single Python version, typically the oldest supported version, as the minimum supported versions of each package may not be compatible with the most recent Python release.

Under the hood, this workflow uses a regular expression to change each `>=` and `~=` specifier in the dependencies to instead be `==`, as pip [does not support](https://github.com/pypa/pip/issues/8085) resolving the minimum versions of packages directly.  Unfortunately, this means that the workflow will only install the minimum version of a package if it is _explicitly_ listed in one of the requirements files with a minimum version.  For instance, if the only listed dependency is `qiskit>=1.0`, this workflow will install `qiskit==1.0` along with the latest version of each transitive dependency, such as `rustworkx`.

## Code coverage (`coverage.yml`)

This workflow tests the [coverage environment](/tests/#coverage-environment) on a single version of Python by installing tox and running `tox -e coverage`.

## Documentation (`docs.yml`)

This workflow ensures that the [Sphinx](https://www.sphinx-doc.org/) documentation builds successfully.  It also publishes the resulting build to [GitHub Pages](https://pages.github.com/) if it is from the appropriate branch (e.g., `main`).

## Citation preview (`citation.yml`)

This workflow is only triggered when the `CITATION.bib` file is changed.  It ensures that the file contains only ASCII characters ([escaped codes](https://en.wikibooks.org/wiki/LaTeX/Special_Characters#Escaped_codes) are preferred, as then the `bib` file will work even when `inputenc` is not used).  It also compiles a sample LaTeX document which includes the citation in its bibliography and uploads the resulting PDF as an artifact so it can be previewed (e.g., before merging a pull request).

## Release (`release.yml`)

This workflow is triggered by a maintainer pushing a tag that represents a release.  It publishes the release to github.com and to [PyPI](https://pypi.org/).

## Docker (`docker.yml`)

This workflow runs periodically (weekly, at the time of writing) to ensure that the [`Dockerfile`](/Dockerfile) and [`compose.yaml`](/compose.yaml) files at the root of the repository result in a successful build with notebooks that execute without error.
