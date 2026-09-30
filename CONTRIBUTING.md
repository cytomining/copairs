# Contributing to copairs

Thank you for your interest in contributing to `copairs`! Bug reports, documentation improvements, tests, fixes, and new features are welcome.

## Before you start

Check the existing issues and pull requests to avoid duplicating work. For substantial changes, please open an issue first so the approach can be discussed with the maintainers.

## Development setup

Fork and clone the repository:

```bash
git clone git@github.com:YOUR_USERNAME/copairs.git
cd copairs
```

### Without Nix

Nix is optional. With Python 3.9 or newer and [uv](https://docs.astral.sh/uv/) installed, create the development environment and install [Prek](https://prek.j178.dev/), the portable hook runner:

```bash
uv sync --frozen --all-extras --dev
uv tool install prek
```

Alternatively, use Python's built-in virtual environment support and pip; neither uv nor Nix is required:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]" prek
```

The pip command installs the package in editable mode with test dependencies; add the `demo` or `docs` extras when working on examples or documentation.

### With Nix

```bash
nix develop
```

This provides the development environment, installs the frozen Python dependencies, and installs the portable Git hook automatically.

## Making changes

Create a focused branch and follow the conventions already used in the codebase. Add or update tests for behavioral changes and use NumPy-style docstrings for public APIs.

Before opening a pull request, run the tests:

```bash
uv run python -m pytest -s
```

If you used pip instead of uv, run `python -m pytest -s` in the activated virtual environment.

### Linting and formatting without Nix

Run the portable Ruff hooks manually; installing a Git hook is not required:

```bash
prek run --all-files
```

Prek reads the committed `.pre-commit-config.yaml` and downloads the pinned Ruff version into an isolated environment on its first run (internet access is required). This is the same Ruff version used by CI: you do not need Nix or a separate Ruff installation, and you do not need to regenerate the configuration.

The hooks apply automatic fixes and formatting. Review the changes with `git diff` and rerun the command if files were changed. To run these checks automatically at commit time, optionally run:

```bash
prek install
```

Contributors already using pre-commit can use `pre-commit run --all-files` and optionally `pre-commit install` with the same configuration. Prefer these pinned hooks over an independently installed Ruff, which may have a different version.

### Additional Nix checks

CI also checks Nix formatting and that the generated portable configuration is up to date. Nix users can run these non-mutating checks locally without building the full development environment:

```bash
nix build --no-link --print-build-logs .#checks.x86_64-linux.formatting .#checks.x86_64-linux.pre-commit-config
```

Replace `x86_64-linux` with your Nix system on other platforms. To apply fixes and formatting, run `nix fmt`.

The portable configuration is generated, not hand-edited. After intentionally updating the Nix tooling, a Nix user or maintainer should regenerate and commit it:

```bash
nix run .#update-pre-commit-config
```

## Submitting a pull request

Use clear, descriptive commits and keep each pull request focused on one change. In the pull request description:

- explain what changed and why;
- link any related issues;
- describe the tests you ran; and
- note any compatibility or performance implications.

All CI checks must pass before a pull request can be merged. If you have questions, open an issue and the maintainers will be happy to help.
