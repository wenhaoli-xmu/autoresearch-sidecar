# autoresearch-sidecar

`autoresearch-sidecar` is a small sidecar agent for Karpathy-style `autoresearch` repositories.

It is not a replacement for the target training repo. Instead, it attaches to an existing repo, reads its `train.py` / `prepare.py` contract, writes isolated experiment variants under `namespace/<node_id>/`, runs them, and keeps iterating.

## What It Is

- `runtime.py`: generic role / phase / tool runtime
- `bundle.py`: target-repo adapter and contract manifest
- `world.py`: experiment execution backend for a target repo
- `program.py`: autoresearch planner + implementer workflow
- `main.py`: CLI entrypoint

## Target Repo Assumptions

The default bundle is built for the Karpathy `autoresearch` training shape:

- the target repo has a root-level `train.py`
- the target repo has a root-level `prepare.py`
- the metric is printed as `val_bpb:`
- peak memory is printed as `peak_vram_mb:`

The current implementation protects against the implementer silently switching to a different project layout or dataset convention.

## Quick Start

Requirements:

- Python 3.10+
- `uv`
- an OpenRouter-compatible API key in `.env` or the shell environment
- a prepared target repo with data/tokenizer already available

Create `.env` from the template:

```bash
cp .env.example .env
```

Run against a target repo:

```bash
uv run python3 main.py --repo-root /path/to/autoresearch-repo
```

If `autoresearch-sidecar` itself is the current repo root:

```bash
uv run python3 main.py
```

Useful flags:

```bash
uv run python3 main.py --help
```

## Example

Run one debug iteration against a target repo:

```bash
uv run python3 main.py \
  --repo-root /path/to/autoresearch-repo \
  --max-iterations 1 \
  --debug
```

## Environment

`main.py` loads environment variables in this order:

1. existing shell environment
2. `<repo-root>/.env`
3. `<cwd>/.env` if different from `<repo-root>`

Shell environment variables win.

Relevant variables:

- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL`
- `AUTORESEARCH_MODEL`

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .env.example
├── main.py
├── runtime.py
├── bundle.py
├── world.py
└── program.py
```

## License

MIT
