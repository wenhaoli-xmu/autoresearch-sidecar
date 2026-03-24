from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from bundle import WorldConfig


class IdeaStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Idea:
    node_id: str
    parent_id: Optional[str]
    illustration: str
    tldr: str
    metric: float | None = None
    peak_vram_mb: float | None = None
    memory_gb: float | None = None
    exit_code: int | None = None
    status: IdeaStatus = IdeaStatus.PENDING

    def as_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "illustration": self.illustration,
            "tldr": self.tldr,
            "metric": self.metric,
            "peak_vram_mb": self.peak_vram_mb,
            "memory_gb": self.memory_gb,
            "exit_code": self.exit_code,
            "status": self.status.value,
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)


class Executor:
    def __init__(self, gpu_ids: list[int | None] | None) -> None:
        slots = gpu_ids or [None]
        self.available_gpus: asyncio.Queue[int | None] = asyncio.Queue()
        for gpu_id in slots:
            self.available_gpus.put_nowait(gpu_id)

    async def execute(self, world: "ResearchWorld", idea: Idea) -> None:
        gpu_id = await self.available_gpus.get()
        idea_dir = world.node_dir(idea.node_id)
        stdout_path = idea_dir / "stdout.log"
        stderr_path = idea_dir / "stderr.log"
        script_path = idea_dir / world.code_filename

        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        pythonpath_parts = [str(world.repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        command = [*world.runner_command, os.path.relpath(script_path, world.repo_root)]
        gpu_label = "default" if gpu_id is None else str(gpu_id)
        print(f"Executing {idea.node_id} on slot {gpu_label}", flush=True)

        try:
            world.update_idea(idea.node_id, status=IdeaStatus.RUNNING)
            with stdout_path.open("w") as f_out, stderr_path.open("w") as f_err:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(world.repo_root),
                    env=env,
                    stdout=f_out,
                    stderr=f_err,
                )
                await proc.wait()

            summary = world.extract_summary(idea.node_id)
            world.update_idea(
                idea.node_id,
                exit_code=proc.returncode,
                metric=summary["metric"],
                peak_vram_mb=summary["peak_vram_mb"],
                memory_gb=summary["memory_gb"],
                status=IdeaStatus.SUCCESS if proc.returncode == 0 and summary["metric"] is not None else IdeaStatus.FAILED,
            )
        except Exception:
            world.update_idea(idea.node_id, status=IdeaStatus.FAILED)
        finally:
            self.available_gpus.put_nowait(gpu_id)


class ResearchWorld:
    def __init__(self, config: WorldConfig, gpu_ids: list[int | None] | None = None) -> None:
        self.config = config
        self.repo_root = config.repo_root
        self.namespace = config.namespace_dir
        self.init_code_path = config.init_code_path
        self.runner_command = config.runner_command
        self.code_filename = config.code_filename
        self.readable_files = dict(config.readable_files)
        self.metric_re = re.compile(config.metric_pattern, re.MULTILINE)
        self.peak_vram_re = re.compile(config.peak_vram_pattern, re.MULTILINE)
        self.nodes: dict[str, Idea] = {}
        self.executor = Executor(gpu_ids)
        self.root_id: str | None = None

    def initialize(self, clear: bool = True) -> Idea:
        if clear and self.namespace.exists():
            shutil.rmtree(self.namespace)
        self.namespace.mkdir(parents=True, exist_ok=True)
        self.nodes = {}
        self.root_id = None

        root = Idea(
            node_id=self.new_node_id(),
            parent_id=None,
            illustration="baseline",
            tldr="baseline",
            status=IdeaStatus.PENDING,
        )
        self.nodes[root.node_id] = root
        self.root_id = root.node_id
        self.persist_idea(root)
        shutil.copy(self.init_code_path, self.node_dir(root.node_id) / self.code_filename)
        return root

    def new_node_id(self) -> str:
        return str(uuid.uuid4())[:6]

    def node_dir(self, node_id: str) -> Path:
        path = self.namespace / node_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def persist_idea(self, idea: Idea) -> None:
        (self.node_dir(idea.node_id) / "meta.json").write_text(idea.as_json())

    def update_idea(self, node_id: str, **updates) -> Idea:
        idea = self.nodes[node_id]
        for key, value in updates.items():
            setattr(idea, key, value)
        self.persist_idea(idea)
        return idea

    def add_idea(self, *, parent_id: str | None, tldr: str, illustration: str) -> Idea:
        idea = Idea(
            node_id=self.new_node_id(),
            parent_id=parent_id,
            tldr=tldr,
            illustration=illustration,
            status=IdeaStatus.PENDING,
        )
        self.nodes[idea.node_id] = idea
        self.persist_idea(idea)
        return idea

    def has_code(self, node_id: str) -> bool:
        return (self.node_dir(node_id) / self.code_filename).exists()

    def pending_nodes(self) -> list[Idea]:
        return [node for node in self.nodes.values() if node.status == IdeaStatus.PENDING]

    async def run_pending_experiments(self) -> None:
        pending = self.pending_nodes()
        if not pending:
            return
        await asyncio.gather(*(self.executor.execute(self, node) for node in pending))

    def best_success(self) -> Idea | None:
        candidates = [node for node in self.nodes.values() if node.status == IdeaStatus.SUCCESS and node.metric is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda node: node.metric)

    def snapshot(self) -> str:
        roots = [nid for nid, node in self.nodes.items() if not node.parent_id or node.parent_id not in self.nodes]
        lines: list[str] = []

        def traverse(nid: str, indent: str = "", visited: set[str] | None = None) -> None:
            seen = visited or set()
            if nid in seen:
                return
            seen.add(nid)

            node = self.nodes[nid]
            node_info = (
                f"ID: {node.node_id} | ParentID: {node.parent_id} | TLDR: {node.tldr} | "
                f"Status: {node.status.value} | val_bpb: {node.metric} | mem_gb: {node.memory_gb}"
            )
            lines.append(f"{indent}* {node_info}")

            children = sorted((cid for cid, child in self.nodes.items() if child.parent_id == nid))
            for index, child_id in enumerate(children):
                is_last = index == len(children) - 1
                child_indent = indent + ("  " if is_last else "| ")
                traverse(child_id, child_indent, seen)

        for root_id in sorted(roots):
            traverse(root_id)
        return "\n".join(lines)

    def extract_summary(self, node_id: str) -> dict[str, float | None]:
        stdout = self.read_stdout(node_id)
        metric_match = self.metric_re.search(stdout)
        peak_match = self.peak_vram_re.search(stdout)
        peak_vram_mb = float(peak_match.group(1)) if peak_match else None
        return {
            "metric": float(metric_match.group(1)) if metric_match else None,
            "peak_vram_mb": peak_vram_mb,
            "memory_gb": round(peak_vram_mb / 1024.0, 1) if peak_vram_mb is not None else None,
        }

    def write_code(self, node_id: str, code: str) -> None:
        self._write_node_file(node_id, self.code_filename, code)

    def mark_failed(self, node_id: str, message: str, exit_code: int = 1) -> None:
        self._write_node_file(node_id, "stderr.log", message.rstrip() + "\n")
        stdout_path = self.node_dir(node_id) / "stdout.log"
        if not stdout_path.exists():
            stdout_path.write_text("")
        self.update_idea(node_id, status=IdeaStatus.FAILED, exit_code=exit_code)

    def read_meta(self, node_id: str) -> str:
        return self._read_node_file(node_id, "meta.json")

    def read_code(self, node_id: str) -> str:
        return self._read_node_file(node_id, self.code_filename)

    def read_stdout(self, node_id: str) -> str:
        return self._read_node_file(node_id, "stdout.log", missing_ok=True)

    def read_stderr(self, node_id: str) -> str:
        return self._read_node_file(node_id, "stderr.log", missing_ok=True)

    def make_tool_handlers(self) -> dict[str, Callable[[str], str]]:
        handlers: dict[str, Callable[[str], str]] = {}
        for tool_name, relative_path in self.readable_files.items():
            handlers[tool_name] = self._make_reader(relative_path)
        return handlers

    def _make_reader(self, relative_path: str) -> Callable[[str], str]:
        def reader(node_id: str) -> str:
            return self._read_node_file(node_id, relative_path, missing_ok=True)

        return reader

    def _read_node_file(self, node_id: str, relative_path: str, missing_ok: bool = False) -> str:
        path = self.node_dir(node_id) / relative_path
        if missing_ok and not path.exists():
            return ""
        return path.read_text()

    def _write_node_file(self, node_id: str, relative_path: str, contents: str) -> None:
        (self.node_dir(node_id) / relative_path).write_text(contents)
