# Test environments

This repository's tests and development automation tasks are organized using [tox], a command-line CI frontend for Python projects.  tox is typically used during local development and is also invoked from this repository's GitHub Actions [workflows](../.github/workflows/).

tox can be installed by running `pip install tox`.

tox is organized around various "environments," each of which is described below.  To run _all_ test environments, run `tox` without any arguments:

```sh
$ tox
```

Environments for this repository are configured in [`tox.ini`] as described below.

## Lint environment

The `lint` environment ensures that the code meets basic coding standards, including

- Formatting and style checking with [ruff], as configured in [`pyproject.toml`]
- Verification that every source file carries the project's copyright header
- [mypy] type annotation checker
- [pylint] checks for a small set of additional rules
- Alt-text validation of images in docstrings via `sphinx-alt-text-validator`

The ruff and pylint passes are applied also to [Jupyter] notebooks under [`docs/`](../docs/) (via [nbqa]).  Notebooks are linted and format-checked but are not executed.

To run:

```sh
$ tox -e lint
```

## Style environment

The command `tox -e style` will apply automated style fixes.  This includes:

- Reformatting of all files in the repository according to [ruff] style
- Automated lint fixes from [ruff], including inside Jupyter notebooks (via [nbqa])

## Test (py##) environments

The `py##` environments are the main test environments.  tox defines one for each version of Python.  For instance, the following command will run the tests on Python 3.10, Python 3.11, and Python 3.12:

```sh
$ tox -e py310,py311,py312
```

These environments execute all tests using [pytest], which supports its own simple style of tests, in addition to [unittest]-style tests.

## Optional dependencies

Some features need an optional dependency, which is installed through an extra (`pip install qiskit-mitigation[slc]`; the `all` extra pulls in every one of them).  Each optional dependency has a lazy import tester and every test that needs one is guarded with it:

```python
from qiskit_mitigation.pec import HAS_SLC


@pytest.mark.skipif(not HAS_SLC, reason="qiskit_addon_slc not installed")
def test_something_traced(): ...
```

The `py##` environments deliberately install none of the optional dependencies.  Running them therefore exercises everything with the optional dependencies absent, so a test that forgot its guard fails rather than passing unnoticed.  Expect the guarded tests to skip there; a _failure_ means a missing guard.

The `optional` and `py##-optional` environments are the exact complement: they install the `all` extra and select every test that those guards skip (`pytest -m skipif`), so they get exercised with the dependencies present.

```sh
$ tox -e py310-optional
```

Since the selection is by the `skipif` marker, guarding a new test is all it takes to have it selected; there is no list of tests or files to maintain.  For the same reason, a `skipif` that is unrelated to an optional dependency gets selected too, which is harmless: such a test merely runs a second time.

The [coverage environment](#coverage-environment) also installs the `all` extra, so that the code paths behind the optional dependencies count towards the coverage.

## Coverage environment

The `coverage` environment uses [Coverage.py] to ensure that the fraction of code tested by pytest is above some threshold (enforced to be 100% for new modules).  A detailed, line-by-line coverage report can be viewed by navigating to `htmlcov/index.html` in a web browser.

To run:

```sh
$ tox -e coverage
```

## Documentation environment

The `docs` environment builds the [Sphinx] documentation locally.

For the documentation build to succeed, [pandoc](https://pandoc.org/) must be installed.  Pandoc is not available via pip, so must be installed through some other means.  Linux users are encouraged to install it through their package manager (e.g., `sudo apt-get install -y pandoc`), while macOS users are encouraged to install it via [Homebrew](https://brew.sh/) (`brew install pandoc`).  Full instructions are available on [pandoc's installation page](https://pandoc.org/installing.html).

To run this environment:

```sh
$ tox -e docs
```

If the build succeeds, it can be viewed by navigating to `docs/_build/html/index.html` in a web browser.

The `docs-clean` environment removes the generated documentation build and autosummary stubs:

```sh
$ tox -e docs-clean
```

[tox]: https://github.com/tox-dev/tox
[`tox.ini`]: ../tox.ini
[`pyproject.toml`]: ../pyproject.toml
[mypy]: https://mypy.readthedocs.io/en/stable/
[ruff]: https://github.com/astral-sh/ruff
[pylint]: https://github.com/PyCQA/pylint
[nbqa]: https://github.com/nbQA-dev/nbQA
[Jupyter]: https://jupyter.org/
[pytest]: https://docs.pytest.org/
[unittest]: https://docs.python.org/3/library/unittest.html
[Coverage.py]: https://coverage.readthedocs.io/
[Sphinx]: https://www.sphinx-doc.org/
