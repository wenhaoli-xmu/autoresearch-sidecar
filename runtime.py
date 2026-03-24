from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import requests


JsonDict = dict[str, Any]
StateDict = dict[str, Any]
Parser = Callable[[str], Any]
Validator = Callable[[Any], Any]
ToolHandler = Callable[[str], str]


@dataclass(frozen=True)
class ToolSpec:
    signature: str
    description: str

    def render(self) -> str:
        return f"{self.signature}: {self.description}"


@dataclass(frozen=True)
class OutputSpec:
    instructions: str
    parser: Parser
    validator: Validator | None = None


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    purpose: str
    reads: tuple[str, ...]
    writes: str
    instructions: str
    output: OutputSpec
    allow_tools: bool = False
    allowed_tools: tuple[str, ...] = ()
    max_tool_rounds: int = 8


@dataclass(frozen=True)
class RoleSpec:
    name: str
    purpose: str
    system_context: str
    required_inputs: tuple[str, ...]
    field_descriptions: Mapping[str, str]
    phases: tuple[PhaseSpec, ...]
    invariants: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    argument: str
    raw_text: str


class ChatCompletionClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 300) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[JsonDict], stop: list[str] | None = None) -> str:
        payload: JsonDict = {
            "model": self.model,
            "messages": messages,
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "choices" not in data:
            raise RuntimeError(f"API error: {data}")
        message = data["choices"][0]["message"]
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
        raise RuntimeError(f"Model returned non-text content: {message}")


class ToolHost:
    TOOL_CALL_RE = re.compile(
        r"<tool>\s*([A-Za-z_][A-Za-z0-9_]*)\(\s*[\"'](.*?)[\"']\s*\)\s*</tool>",
        re.DOTALL,
    )

    def __init__(
        self,
        tool_specs: Mapping[str, ToolSpec],
        handlers: Mapping[str, ToolHandler],
    ) -> None:
        self.tool_specs = dict(tool_specs)
        self.handlers = dict(handlers)

    def render_protocol(self, allowed_tools: tuple[str, ...]) -> str:
        lines = ["Allowed tools for this phase:"]
        for name in allowed_tools:
            spec = self.tool_specs[name]
            lines.append(f"- {spec.render()}")
        lines.extend(
            (
                "",
                "Tool protocol:",
                "- Emit exactly one tool call wrapped as <tool>...</tool><stop>.",
                "- Only one tool call is allowed per assistant turn.",
                "- After the host injects <result>...</result>, continue the same phase.",
                "- When investigation is complete, stop calling tools and emit the required phase output.",
            )
        )
        return "\n".join(lines)

    def parse(self, raw_text: str) -> ToolInvocation | None:
        match = self.TOOL_CALL_RE.search(raw_text)
        if not match:
            return None
        return ToolInvocation(
            name=match.group(1),
            argument=match.group(2),
            raw_text=match.group(0).strip(),
        )

    def execute(self, tool_name: str, argument: str, allowed_tools: tuple[str, ...]) -> str:
        if tool_name not in allowed_tools:
            return f"Error: tool '{tool_name}' is not allowed in this phase."
        handler = self.handlers.get(tool_name)
        if handler is None:
            return f"Error: unknown tool '{tool_name}'."
        try:
            result = handler(argument)
        except Exception as exc:  # pragma: no cover - defensive host boundary
            return f"Error: {tool_name}({argument!r}) failed with {exc!r}."
        return result if isinstance(result, str) else json.dumps(result, indent=2, ensure_ascii=False)


class RoleRunner:
    TOOL_BLOCK_RE = re.compile(
        r"(.*?<tool>\s*[A-Za-z_][A-Za-z0-9_]*\(\s*[\"'].*?[\"']\s*\)\s*</tool>\s*<stop>)",
        re.DOTALL,
    )

    def __init__(self, client: ChatCompletionClient, tool_host: ToolHost, debug_mode: bool = False) -> None:
        self.client = client
        self.tool_host = tool_host
        self.debug_mode = debug_mode

    def run(self, role: RoleSpec, initial_state: StateDict) -> StateDict:
        missing = [name for name in role.required_inputs if name not in initial_state]
        if missing:
            raise ValueError(f"Missing required inputs for role {role.name}: {missing}")

        state = dict(initial_state)
        for phase in role.phases:
            raw_output = self._run_phase(role, phase, state)
            parsed = phase.output.parser(raw_output)
            if phase.output.validator is not None:
                parsed = phase.output.validator(parsed)
            state[phase.writes] = parsed
        return state

    def _run_phase(self, role: RoleSpec, phase: PhaseSpec, state: StateDict) -> str:
        history: list[JsonDict] = [
            {"role": "system", "content": self._render_system_prompt(role, phase)},
            {"role": "user", "content": self._render_user_prompt(role, phase, state)},
        ]

        tool_rounds = 0
        while True:
            raw = self.client.complete(history, stop=["<stop>"] if phase.allow_tools else None)
            normalized = raw
            if phase.allow_tools and "</tool>" in raw and "<stop>" not in raw:
                normalized = f"{raw}<stop>"

            match = self.TOOL_BLOCK_RE.search(normalized) if phase.allow_tools else None
            if match is None:
                history.append({"role": "assistant", "content": raw})
                if self.debug_mode:
                    print(f"\n[phase {phase.name} output]\n{raw}\n", flush=True)
                return raw

            if tool_rounds >= phase.max_tool_rounds:
                raise RuntimeError(f"Phase {phase.name} exceeded its tool budget ({phase.max_tool_rounds}).")

            assistant_message = match.group(1).strip()
            history.append({"role": "assistant", "content": assistant_message})

            invocation = self.tool_host.parse(assistant_message)
            if invocation is None:
                raise ValueError(f"Malformed tool call in phase {phase.name}: {assistant_message}")

            tool_result = self.tool_host.execute(invocation.name, invocation.argument, phase.allowed_tools)
            history.append({"role": "user", "content": f"<result>\n{tool_result}\n</result>"})
            tool_rounds += 1

            if self.debug_mode:
                print(f"\n[tool {invocation.name}({invocation.argument!r})]\n{tool_result}\n", flush=True)

    def _render_system_prompt(self, role: RoleSpec, phase: PhaseSpec) -> str:
        lines = [
            role.system_context.strip(),
            f"Role: {role.name}",
            f"Role purpose: {role.purpose}",
            f"Current phase: {phase.name}",
            f"Phase purpose: {phase.purpose}",
            "Phase instructions:",
            phase.instructions.strip(),
        ]
        if role.invariants:
            lines.append("Role invariants:")
            lines.extend(f"- {item}" for item in role.invariants)
        if phase.allow_tools:
            lines.append(self.tool_host.render_protocol(phase.allowed_tools))
        lines.extend(("Output contract:", phase.output.instructions.strip()))
        return "\n".join(lines)

    def _render_user_prompt(self, role: RoleSpec, phase: PhaseSpec, state: StateDict) -> str:
        lines = ["Phase state bindings:"]
        for field_name in phase.reads:
            description = role.field_descriptions.get(field_name, "")
            if description:
                lines.append(f"\n[{field_name}] {description}")
            else:
                lines.append(f"\n[{field_name}]")
            lines.append(self._serialize(state.get(field_name)))
        lines.append("\nEmit the required phase output now.")
        return "\n".join(lines)

    def _serialize(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
