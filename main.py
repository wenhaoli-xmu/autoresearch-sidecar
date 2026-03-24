from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from bundle import make_karpathy_autoresearch_bundle
from program import AutoresearchProgram
from runtime import ChatCompletionClient, RoleRunner, ToolHost
from world import IdeaStatus, ResearchWorld


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def load_default_envs(repo_root: Path) -> None:
    load_env_file(repo_root / ".env")
    cwd_env = Path.cwd().resolve() / ".env"
    if cwd_env != (repo_root / ".env").resolve():
        load_env_file(cwd_env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoresearch-sidecar against a target repo.")
    parser.add_argument("--repo-root", default=".", help="Path to the target autoresearch repo.")
    parser.add_argument("--namespace-dir", default=None, help="Override namespace directory inside the target repo.")
    parser.add_argument("--max-iterations", type=int, default=10, help="Number of planner/implementer iterations to run after baseline.")
    parser.add_argument("--debug", action="store_true", help="Print intermediate role/tool outputs.")
    parser.add_argument("--model", default=None, help="Override the LLM model name.")
    parser.add_argument("--base-url", default=None, help="Override the chat completions base URL.")
    parser.add_argument("--gpu-id", dest="gpu_ids", action="append", type=int, default=None, help="Repeat to provide one or more GPU ids.")
    return parser.parse_args()


async def run_autoresearch_loop(
    repo_root: str | Path = ".",
    *,
    namespace_dir: str | Path | None = None,
    max_iterations: int = 10,
    gpu_ids: list[int | None] | None = None,
    debug: bool = False,
    model: str | None = None,
    base_url: str | None = None,
) -> None:
    repo_root = Path(repo_root).resolve()
    load_default_envs(repo_root)

    bundle = make_karpathy_autoresearch_bundle(repo_root=repo_root, namespace_dir=namespace_dir)
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    world = ResearchWorld(bundle.world, gpu_ids=gpu_ids)
    root = world.initialize(clear=True)

    client = ChatCompletionClient(
        base_url=base_url or os.getenv("OPENROUTER_BASE_URL", bundle.program.default_base_url),
        api_key=api_key,
        model=model or os.getenv("AUTORESEARCH_MODEL", bundle.program.default_model),
    )
    tool_host = ToolHost(bundle.program.tool_specs, world.make_tool_handlers())
    runner = RoleRunner(client, tool_host, debug_mode=debug)
    program = AutoresearchProgram(bundle, runner)

    print(f"Target repo: {repo_root}", flush=True)
    print(f"Namespace: {bundle.world.namespace_dir}", flush=True)
    print("Running initial baseline (root node)...", flush=True)
    await world.run_pending_experiments()
    if world.nodes[root.node_id].status == IdeaStatus.FAILED:
        print("Root node failed, exiting.")
        return

    for index in range(max_iterations):
        print(f"\n{'=' * 20} Iteration {index + 1} / {max_iterations} {'=' * 20}", flush=True)
        new_node_ids = await program.run_iteration(world)
        print(f"New nodes: {new_node_ids}", flush=True)
        print("\nCurrent research tree:", flush=True)
        print(world.snapshot(), flush=True)


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_autoresearch_loop(
            repo_root=args.repo_root,
            namespace_dir=args.namespace_dir,
            max_iterations=args.max_iterations,
            gpu_ids=args.gpu_ids,
            debug=args.debug,
            model=args.model,
            base_url=args.base_url,
        )
    )


if __name__ == "__main__":
    main()
