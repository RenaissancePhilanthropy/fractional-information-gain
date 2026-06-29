# Contributing

## Development

The project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync --dev --extra torch
uv run pytest
```

To build the docs locally:

```bash
uv sync --dev --extra docs --extra torch
uv run mkdocs serve
```

## Releasing

Releases are published to PyPI by the `.github/workflows/publish.yml` workflow using
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/). To make a release,
bump `version` in `pyproject.toml`, then publish a GitHub Release tagged `vX.Y.Z`. The
workflow builds the sdist/wheel and uploads them to PyPI. Running the workflow manually
uploads to TestPyPI for a dry run.
