from __future__ import annotations

import json
import re
from typing import Any

from bundle import ResearchBundle
from runtime import OutputSpec, PhaseSpec, RoleRunner, RoleSpec
from world import ResearchWorld


def parse_text(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"Expected text output, got {type(raw).__name__}.")
    return raw.strip()


def validate_nonempty_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected non-empty text output.")
    return value.strip()


def parse_json_list(raw: str) -> Any:
    if not isinstance(raw, str):
        raise ValueError(f"Expected JSON text output, got {type(raw).__name__}.")
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def validate_proposals(value: Any) -> list[dict[str, str | None]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a non-empty list of proposals.")

    cleaned: list[dict[str, str | None]] = []
    for item in value[:3]:
        if not isinstance(item, dict):
            raise ValueError(f"Each proposal must be an object, got {type(item)!r}.")
        parent_id = item.get("parent_id")
        tldr = item.get("tldr")
        illustration = item.get("illustration")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError("parent_id must be a string or null.")
        if not isinstance(tldr, str) or not tldr.strip():
            raise ValueError("Proposal tldr must be non-empty.")
        if not isinstance(illustration, str) or not illustration.strip():
            raise ValueError("Proposal illustration must be non-empty.")
        cleaned.append(
            {
                "parent_id": parent_id.strip() if isinstance(parent_id, str) else None,
                "tldr": tldr.strip(),
                "illustration": illustration.strip(),
            }
        )
    return cleaned


def validate_python_source(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected non-empty Python source.")
    source = value.strip()
    if "val_bpb:" not in source or "peak_vram_mb:" not in source:
        raise ValueError("Generated train.py does not preserve the required summary lines.")
    return source


def validate_train_py_against_parent(bundle: ResearchBundle, parent_source: str, candidate_source: str) -> str:
    missing = [
        anchor
        for anchor in bundle.program.required_parent_anchors
        if anchor in parent_source and anchor not in candidate_source
    ]
    if missing:
        raise ValueError(
            "Generated train.py dropped required parent-contract anchors: "
            + ", ".join(repr(item) for item in missing)
        )

    introduced = [
        pattern
        for pattern in bundle.program.forbidden_new_patterns
        if re.search(pattern, candidate_source) and not re.search(pattern, parent_source)
    ]
    if introduced:
        raise ValueError(
            "Generated train.py introduced incompatible data/config assumptions: "
            + ", ".join(repr(item) for item in introduced)
        )
    return candidate_source


def build_planner_role(bundle: ResearchBundle) -> RoleSpec:
    tool_names = tuple(bundle.program.tool_specs.keys())
    return RoleSpec(
        name="planner",
        purpose="Propose the next batch of autoresearch train.py experiments.",
        system_context=f"{bundle.program.research_world_context}\n\n{bundle.program.contract}",
        required_inputs=("snapshot",),
        field_descriptions={
            "snapshot": "Serialized research tree snapshot.",
            "research_notes": "Concise findings from tool-assisted inspection.",
            "proposals": "A JSON list of proposal objects with parent_id, tldr, and illustration.",
        },
        invariants=(
            "Investigation and proposal emission are separate phases.",
            "Use tools to ground proposals in existing code or results.",
            "Prefer improving strong successful nodes or debugging informative failures.",
            "Do not emit markdown fences or conversational filler in final outputs.",
        ),
        phases=(
            PhaseSpec(
                name="investigate",
                purpose="Inspect promising nodes and collect explicit research notes.",
                reads=("snapshot",),
                writes="research_notes",
                instructions="""
Inspect successful nodes and informative failures before proposing new work.
Use tools one call at a time.
Focus on ideas that could improve val_bpb without violating the train.py-only contract.
Output concise research notes only. Do not emit proposals yet.
""".strip(),
                output=OutputSpec(
                    instructions="Return only concise research notes. No JSON. No markdown fences.",
                    parser=parse_text,
                    validator=validate_nonempty_text,
                ),
                allow_tools=True,
                allowed_tools=tool_names,
                max_tool_rounds=6,
            ),
            PhaseSpec(
                name="emit_proposals",
                purpose="Turn the research notes into concrete experimental proposals.",
                reads=("snapshot", "research_notes"),
                writes="proposals",
                instructions="""
Return 1 to 3 concrete proposals.
Each proposal must include parent_id, tldr, and illustration.
Choose parent_id values from the snapshot when possible.
Output only the JSON list.
""".strip(),
                output=OutputSpec(
                    instructions='Return only valid JSON of the form [{"parent_id": "...", "tldr": "...", "illustration": "..."}].',
                    parser=parse_json_list,
                    validator=validate_proposals,
                ),
            ),
        ),
    )


def build_implementer_role(bundle: ResearchBundle) -> RoleSpec:
    tool_names = tuple(bundle.program.tool_specs.keys())
    return RoleSpec(
        name="implementer",
        purpose="Implement one autoresearch proposal as a runnable train.py variant.",
        system_context=f"{bundle.program.research_world_context}\n\n{bundle.program.contract}",
        required_inputs=("snapshot", "node_id", "parent_id", "tldr", "illustration", "parent_source"),
        field_descriptions={
            "snapshot": "Serialized research tree snapshot.",
            "node_id": "Node being implemented.",
            "parent_id": "Parent node whose train.py should be used as the baseline.",
            "tldr": "Short summary of the proposal.",
            "illustration": "Detailed rationale for the proposal.",
            "parent_source": "Exact train.py source from the parent node. Treat this as the file you are editing.",
            "implementation_plan": "A concise plan describing the concrete train.py edits.",
            "train_py": "Full runnable Python source for train.py.",
        },
        invariants=(
            "Inspect the parent node before writing code.",
            "Keep the implementation grounded in the chosen parent train.py.",
            "Edit the parent file instead of replacing the task with a different training script.",
            "Only emit the final train.py source in the final phase.",
            "Preserve the fixed summary output contract used by the world.",
        ),
        phases=(
            PhaseSpec(
                name="inspect_parent",
                purpose="Study the parent implementation and prepare an edit plan.",
                reads=("snapshot", "node_id", "parent_id", "tldr", "illustration", "parent_source"),
                writes="implementation_plan",
                instructions="""
Inspect the parent node's code and logs as needed.
Work out the smallest coherent implementation that matches the proposal.
Prefer small, targeted edits over unrelated rewrites.
The parent_source shown in state is the file you must edit.
Preserve the parent's prepare.py integration, data loading path, tokenizer usage, and evaluation harness unless the proposal explicitly changes them.
Do not introduce alternate dataset conventions or config-file conventions that are not already present in parent_source.
Output only the implementation plan, not code.
""".strip(),
                output=OutputSpec(
                    instructions="Return only a concise implementation plan. No code fences.",
                    parser=parse_text,
                    validator=validate_nonempty_text,
                ),
                allow_tools=True,
                allowed_tools=tool_names,
                max_tool_rounds=6,
            ),
            PhaseSpec(
                name="emit_train_py",
                purpose="Emit the full runnable train.py source.",
                reads=("snapshot", "node_id", "parent_id", "tldr", "illustration", "parent_source", "implementation_plan"),
                writes="train_py",
                instructions="""
Generate the full runnable train.py source for this node by editing parent_source.
Most of the file should remain identical to parent_source except for the targeted experiment changes.
Do not emit commentary.
Do not emit markdown fences.
The code must remain compatible with the autoresearch training contract.
Keep integration with prepare.py intact.
Do not switch to a different project layout, dataset format, config file scheme, or evaluation harness.
""".strip(),
                output=OutputSpec(
                    instructions="Return only runnable Python source for train.py.",
                    parser=parse_text,
                    validator=validate_python_source,
                ),
            ),
        ),
    )


class AutoresearchProgram:
    def __init__(self, bundle: ResearchBundle, runner: RoleRunner) -> None:
        self.bundle = bundle
        self.runner = runner
        self.planner_role = build_planner_role(bundle)
        self.implementer_role = build_implementer_role(bundle)

    def propose(self, world: ResearchWorld) -> list[dict[str, str | None]]:
        state = self.runner.run(self.planner_role, {"snapshot": world.snapshot()})
        return state["proposals"]

    def materialize_proposals(
        self,
        world: ResearchWorld,
        proposals: list[dict[str, str | None]],
    ) -> list[str]:
        fallback_parent = None
        best = world.best_success()
        if best is not None:
            fallback_parent = best.node_id
        elif world.root_id is not None:
            fallback_parent = world.root_id

        new_node_ids: list[str] = []
        for proposal in proposals:
            parent_id = proposal["parent_id"]
            if parent_id not in world.nodes:
                parent_id = fallback_parent
            node = world.add_idea(
                parent_id=parent_id,
                tldr=str(proposal["tldr"]),
                illustration=str(proposal["illustration"]),
            )
            new_node_ids.append(node.node_id)
        return new_node_ids

    def implement_pending_nodes(self, world: ResearchWorld) -> list[str]:
        pending_ids = [node.node_id for node in world.pending_nodes() if not world.has_code(node.node_id)]
        if not pending_ids:
            return []

        snapshot = world.snapshot()
        for node_id in pending_ids:
            node = world.nodes[node_id]
            parent_id = node.parent_id or world.root_id
            if parent_id is None:
                raise ValueError(f"Node {node_id} has no parent and world has no root.")
            parent_source = world.read_code(parent_id)
            try:
                state = self.runner.run(
                    self.implementer_role,
                    {
                        "snapshot": snapshot,
                        "node_id": node_id,
                        "parent_id": parent_id,
                        "tldr": node.tldr,
                        "illustration": node.illustration,
                        "parent_source": parent_source,
                    },
                )
                candidate_source = state["train_py"]
                world.write_code(node_id, candidate_source)
                validate_train_py_against_parent(self.bundle, parent_source, candidate_source)
            except Exception as exc:
                world.mark_failed(node_id, f"Implementer rejected before execution: {exc}", exit_code=1)
                continue
        return pending_ids

    async def run_iteration(self, world: ResearchWorld) -> list[str]:
        proposals = self.propose(world)
        new_node_ids = self.materialize_proposals(world, proposals)
        self.implement_pending_nodes(world)
        await world.run_pending_experiments()
        return new_node_ids
