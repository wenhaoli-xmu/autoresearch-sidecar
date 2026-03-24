from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from runtime import ToolSpec


@dataclass(frozen=True)
class WorldConfig:
    repo_root: Path
    namespace_dir: Path
    init_code_path: Path
    runner_command: tuple[str, ...]
    code_filename: str
    readable_files: Mapping[str, str]
    metric_pattern: str
    peak_vram_pattern: str


@dataclass(frozen=True)
class ProgramConfig:
    research_world_context: str
    contract: str
    tool_specs: Mapping[str, ToolSpec]
    required_parent_anchors: tuple[str, ...]
    forbidden_new_patterns: tuple[str, ...]
    default_model: str = "openai/gpt-5.2"
    default_base_url: str = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class ResearchBundle:
    name: str
    world: WorldConfig
    program: ProgramConfig


def make_karpathy_autoresearch_bundle(
    repo_root: str | Path,
    namespace_dir: str | Path | None = None,
) -> ResearchBundle:
    repo_root = Path(repo_root).resolve()
    namespace_dir = Path(namespace_dir).resolve() if namespace_dir else (repo_root / "namespace").resolve()

    tool_specs = {
        "read_meta": ToolSpec(
            signature="read_meta(node_id: str) -> str",
            description="Read the metadata JSON for a research node.",
        ),
        "read_code": ToolSpec(
            signature="read_code(node_id: str) -> str",
            description="Read the train.py source for a research node.",
        ),
        "read_stdout": ToolSpec(
            signature="read_stdout(node_id: str) -> str",
            description="Read the stdout log for a research node.",
        ),
        "read_stderr": ToolSpec(
            signature="read_stderr(node_id: str) -> str",
            description="Read the stderr log for a research node.",
        ),
    }

    research_world_context = """
The research world manages isolated train.py variants under namespace/<node_id>/train.py.
Each node is an experiment with:
- node_id: unique node identifier.
- parent_id: the base node this experiment extends.
- illustration: longer technical rationale for the change.
- tldr: short summary of the change.
- metric: val_bpb from the final summary. Smaller is always better.
- peak_vram_mb / memory_gb: memory usage from the final summary.
- exit_code: process return code.
- status: pending, running, success, or failed.
""".strip()

    contract = """
You are operating inside Karpathy's autoresearch pipeline.

Ground rules derived from README.md and program.md:
- The goal is to lower val_bpb. Smaller is always better.
- The training budget is fixed by prepare.py, so experiments must remain comparable under that wall-clock budget.
- Only train.py may change. prepare.py and its evaluation harness are read-only.
- Do not add dependencies or require extra files.
- Preserve the runnable command shape: the generated file must still work as train.py under `uv run`.
- Preserve the final summary format, especially the exact `val_bpb:` and `peak_vram_mb:` lines.
- Simplicity matters. Prefer changes that are easy to justify and easy to keep if they work.
- Experiments run from the repository root so they reuse the real prepare.py, tokenizer, data, and environment.
""".strip()

    return ResearchBundle(
        name="karpathy_autoresearch",
        world=WorldConfig(
            repo_root=repo_root,
            namespace_dir=namespace_dir,
            init_code_path=repo_root / "train.py",
            runner_command=("uv", "run"),
            code_filename="train.py",
            readable_files={
                "read_meta": "meta.json",
                "read_code": "train.py",
                "read_stdout": "stdout.log",
                "read_stderr": "stderr.log",
            },
            metric_pattern=r"^val_bpb:\s*(-?\d+(?:\.\d+)?)\s*$",
            peak_vram_pattern=r"^peak_vram_mb:\s*(-?\d+(?:\.\d+)?)\s*$",
        ),
        program=ProgramConfig(
            research_world_context=research_world_context,
            contract=contract,
            tool_specs=tool_specs,
            required_parent_anchors=(
                "from prepare import",
                "MAX_SEQ_LEN",
                "TIME_BUDGET",
                "Tokenizer.from_directory()",
                "make_dataloader(",
                "evaluate_bpb(",
                "val_bpb:",
                "peak_vram_mb:",
            ),
            forbidden_new_patterns=(
                r"tinyshakespeare",
                r"train\.bin",
                r"val\.bin",
                r"train\.pt",
                r"val\.pt",
                r"meta\.pkl",
                r"train_config\.json",
                r"config\.json",
                r"TRAIN_CONFIG",
            ),
        ),
    )
