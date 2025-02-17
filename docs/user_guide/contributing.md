# Contributing to the code

If you find a problem with the code or have any ideas for new features or improvements, feel free to get involved! We ask that all problems are reported as [Github issues](https://github.com/ben-cassese/jorbit/issues) and all code changes are submitted as [pull requests](https://github.com/ben-cassese/jorbit/pulls) linked to an issue. Please make one branch per feature or bug fix, and be sure to include tests and Google-style [docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods) for any major additions or changes.

``jorbit`` uses [uv](https://docs.astral.sh/uv/) for requirements management. Once you have a local copy of a branch, you should be able to build a local environment by running:

```bash
uv sync
```

which will create a .venv directory in the root of the repository. You can then activate this environment with:

```bash
source .venv/bin/activate
```

Or, to run anything without activating, use

```bash
uv run <command>
```

If submitting a pull request, please be sure to run the tests (either with [tox](https://tox.wiki/en/latest/) or [pytest](https://docs.pytest.org/en/8.1.x/)) and to adhere to the [black](https://black.readthedocs.io/en/stable/) python code style. ``jorbit`` uses Github Actions for continuous integration, so all pull requests will automatically trigger tests and formatting checks.

Additionally, ``jorbit`` uses a few pre-commit hooks for formatting and linting, including Ruff and Codespell. When creating a new local branch, please run the following command to install the hooks:

```bash
pre-commit install
```

Though ``jorbit`` is not affiliated with [Astropy](http://www.astropy.org/), we still ask that all contributors follow the [Astropy Project Code of Conduct](http://www.astropy.org/code_of_conduct.html). If you have any questions or concerns, please feel free to reach out to the maintainer(s).
