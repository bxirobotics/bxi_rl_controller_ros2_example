#!/usr/bin/env python3
import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml


DEFAULT_MANIFESTS = (Path("src/bxi_example_py_elf3/config/release_protection.yaml"),)
DEFAULT_REMOTE_CONFIG = Path("src/remote_controller/config/xbox_default.yaml")
PUBLIC_DEV_ONLY_PATHS = {
    Path("tools/sanitize_release.py"),
    Path("tools/README.md"),
    Path(".github/workflows/sync_public_main.yml"),
}
ROOT_RELATIVE_PREFIXES = {"src", "tools", ".github"}


@dataclass
class ProtectedState:
    name: str
    behaviors: Set[str]
    model_keys: Set[str]
    files: Set[Path]


@dataclass
class ProtectionSpec:
    manifest_path: Path
    state_machine_config: Path
    robot_states: Path
    demo_node: Path
    states: Dict[str, ProtectedState]


@dataclass
class StateMachineResult:
    removed_states: Set[str]
    removed_events: Set[str]
    outputs_to_remove: Set[str]


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file) or {}


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(data, output_file, sort_keys=False, allow_unicode=True)


def string_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    raise TypeError(f"{field_name} must be a string or a list of strings")


def all_values(configs: Iterable[ProtectedState], field_name: str) -> Set[str]:
    values: Set[str] = set()
    for config in configs:
        values.update(getattr(config, field_name))
    return values


def all_files(configs: Iterable[ProtectedState]) -> Set[Path]:
    files: Set[Path] = set()
    for config in configs:
        files.update(config.files)
    return files


def relative_to_root(source_root: Path, path: Path) -> Path:
    return path.resolve().relative_to(source_root.resolve())


def is_root_relative(path: Path) -> bool:
    return bool(path.parts) and path.parts[0] in ROOT_RELATIVE_PREFIXES


def resolve_manifest_path(source_root: Path, manifest_path: Path, value: str) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        candidate = raw_path
    elif is_root_relative(raw_path):
        candidate = source_root / raw_path
    else:
        candidate = manifest_path.parent / raw_path
    return relative_to_root(source_root, candidate)


def default_package_root(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "config":
        return manifest_path.parent.parent
    return manifest_path.parent


def load_protection_spec(source_root: Path, manifest_arg: Path) -> ProtectionSpec:
    manifest_path = manifest_arg if manifest_arg.is_absolute() else source_root / manifest_arg
    manifest_path = manifest_path.resolve()
    manifest_rel = relative_to_root(source_root, manifest_path)
    manifest = load_yaml(manifest_path)

    package_root = default_package_root(manifest_path)
    package_name = package_root.name
    paths = manifest.get("paths") or {}

    default_state_machine = manifest_path.parent / "elf3_state_machine.yaml"
    default_robot_states = package_root / package_name / "robot_states.py"
    default_demo_node = package_root / package_name / "bxi_example_demo.py"

    state_machine_config = (
        resolve_manifest_path(source_root, manifest_path, str(paths["state_machine"]))
        if paths.get("state_machine") else
        relative_to_root(source_root, default_state_machine)
    )
    robot_states = (
        resolve_manifest_path(source_root, manifest_path, str(paths["robot_states"]))
        if paths.get("robot_states") else
        relative_to_root(source_root, default_robot_states)
    )
    demo_node = (
        resolve_manifest_path(source_root, manifest_path, str(paths["demo_node"]))
        if paths.get("demo_node") else
        relative_to_root(source_root, default_demo_node)
    )

    states: Dict[str, ProtectedState] = {}
    for state_name, raw_config in (manifest.get("protected_states") or {}).items():
        field = f"protected_states.{state_name}"
        config = raw_config or {}
        behaviors = set(string_list(config.get("behavior"), f"{field}.behavior"))
        behaviors.update(string_list(config.get("behaviors"), f"{field}.behaviors"))
        states[str(state_name)] = ProtectedState(
            name=str(state_name),
            behaviors=behaviors,
            model_keys=set(string_list(config.get("model_keys"), f"{field}.model_keys")),
            files={
                resolve_manifest_path(source_root, manifest_path, item)
                for item in string_list(config.get("files"), f"{field}.files")
            },
        )

    return ProtectionSpec(
        manifest_path=manifest_rel,
        state_machine_config=state_machine_config,
        robot_states=robot_states,
        demo_node=demo_node,
        states=states,
    )


def copy_release_tree(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_root:
        raise ValueError("output directory must not be the repository root")
    if output_root.exists():
        shutil.rmtree(output_root)

    def ignore(_dir: str, names: List[str]) -> Set[str]:
        ignored = {
            ".git",
            "build",
            "install",
            "log",
            "dist",
            "update",
            ".update",
            "__pycache__",
            ".pytest_cache",
        }
        return {name for name in names if name in ignored}

    shutil.copytree(source_root, output_root, ignore=ignore)


def transition_target(rule: Any) -> Optional[str]:
    if isinstance(rule, str):
        return rule
    if isinstance(rule, dict):
        target = rule.get("to")
        return str(target) if target is not None else None
    return None


def remote_event_output(event_config: Any) -> Optional[str]:
    if isinstance(event_config, dict):
        slot = event_config.get("slot")
        value = event_config.get("value", 1)
    else:
        slot = event_config
        value = 1
    if slot:
        return f"{slot}={value}"
    return None


def infer_events_to_remove(
    path: Path,
    states: Dict[str, Any],
    states_to_remove: Set[str],
) -> Set[str]:
    candidate_events: Set[str] = set()
    target_removed_events: Set[str] = set()
    public_events: Set[str] = set()

    for state_name, state_config in states.items():
        source_removed = state_name in states_to_remove
        transitions = (state_config or {}).get("transitions") or {}
        on_event = transitions.get("on_event")
        if not isinstance(on_event, dict):
            continue

        for event, rule in on_event.items():
            event_name = str(event)
            target = transition_target(rule)
            target_removed = target in states_to_remove
            if source_removed or target_removed:
                candidate_events.add(event_name)
            if target_removed:
                target_removed_events.add(event_name)
            if not source_removed and not target_removed:
                public_events.add(event_name)

    events_to_remove = candidate_events - public_events
    for event_name in sorted(target_removed_events & public_events):
        warn(
            f"{path}: event '{event_name}' is connected to a protected state but "
            "is also used by public states, keeping that event and its input path"
        )

    return events_to_remove


def sanitize_state_machine(path: Path, protected_states: Set[str]) -> StateMachineResult:
    config = load_yaml(path)
    remote_events = config.get("remote_events") or {}
    states = config.get("states") or {}

    states_to_remove = set(protected_states)
    initial_state = config.get("initial_state")
    if initial_state in states_to_remove:
        states_to_remove.remove(str(initial_state))
        warn(f"{path}: initial_state '{initial_state}' is protected, keeping it")

    events_to_remove = infer_events_to_remove(path, states, states_to_remove)

    for event in sorted(events_to_remove):
        if event not in remote_events:
            warn(
                f"{path}: inferred protected event '{event}' is not defined in remote_events"
            )

    output_to_events: Dict[str, Set[str]] = {}
    event_to_output: Dict[str, str] = {}
    for event, event_config in remote_events.items():
        output = remote_event_output(event_config)
        if output is None:
            continue
        event_name = str(event)
        event_to_output[event_name] = output
        output_to_events.setdefault(output, set()).add(event_name)

    outputs_to_remove: Set[str] = set()
    for event in events_to_remove:
        output = event_to_output.get(event)
        if not output:
            continue
        public_users = output_to_events.get(output, set()) - events_to_remove
        if public_users:
            warn(
                f"{path}: output '{output}' is shared by protected event '{event}' and "
                f"public events {sorted(public_users)}, keeping the remote output binding"
            )
            continue
        outputs_to_remove.add(output)

    for event in events_to_remove:
        remote_events.pop(event, None)

    for state in states_to_remove:
        states.pop(state, None)

    for state_name, state_config in states.items():
        transitions = (state_config or {}).get("transitions") or {}
        on_event = transitions.get("on_event")
        if isinstance(on_event, dict):
            for event in list(on_event.keys()):
                target = transition_target(on_event[event])
                if target in states_to_remove or event in events_to_remove:
                    on_event.pop(event, None)

        after_rules = transitions.get("after")
        if isinstance(after_rules, list):
            transitions["after"] = [
                rule for rule in after_rules
                if transition_target(rule) not in states_to_remove
            ]

    write_yaml(path, config)
    return StateMachineResult(states_to_remove, events_to_remove, outputs_to_remove)


def tokens_in_value(value: Any, tokens: Set[str]) -> Set[str]:
    found: Set[str] = set()
    if isinstance(value, str):
        found.update(token for token in tokens if token and token in value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.update(tokens_in_value(key, tokens))
            found.update(tokens_in_value(item, tokens))
    elif isinstance(value, list):
        for item in value:
            found.update(tokens_in_value(item, tokens))
    return found


def contains_token(value: Any, tokens: Set[str]) -> bool:
    return bool(tokens_in_value(value, tokens))


def sanitize_remote_config(path: Path, tokens: Set[str], protected_outputs: Set[str]) -> Set[str]:
    if not path.exists():
        return set()

    config = load_yaml(path)
    kept_tokens: Set[str] = set()

    outputs = config.get("outputs") or {}
    remaining_bindings: List[Any] = []
    for mode in ("level", "edge"):
        bindings = outputs.get(mode)
        if not isinstance(bindings, list):
            continue
        output_bindings: List[Any] = []
        for binding in bindings:
            output = str((binding or {}).get("output", ""))
            binding_tokens = tokens_in_value(binding, tokens)
            if output in protected_outputs:
                continue
            if binding_tokens:
                kept_tokens.update(binding_tokens)
                warn(
                    f"{path}: keeping output binding '{output}' because it is not a "
                    f"protected-only output; tokens={sorted(binding_tokens)}"
                )
            output_bindings.append(binding)
            remaining_bindings.append(binding)
        outputs[mode] = output_bindings

    controls = config.get("controls")
    if isinstance(controls, dict):
        for control in list(controls.keys()):
            control_tokens = tokens_in_value(control, tokens)
            control_tokens.update(tokens_in_value(controls[control], tokens))
            if not control_tokens:
                continue
            if contains_token(remaining_bindings, {str(control)}):
                kept_tokens.update(control_tokens)
                warn(
                    f"{path}: keeping control '{control}' because a remaining output "
                    f"binding still references it"
                )
                continue
            controls.pop(control, None)

    for source_group in (config.get("sources") or {}).values():
        signals = (source_group or {}).get("signals")
        if not isinstance(signals, dict):
            continue
        for signal in list(signals.keys()):
            signal_tokens = tokens_in_value(signal, tokens)
            signal_tokens.update(tokens_in_value(signals[signal], tokens))
            if not signal_tokens:
                continue
            if contains_token(controls or {}, {str(signal)}):
                kept_tokens.update(signal_tokens)
                warn(
                    f"{path}: keeping signal '{signal}' because a remaining control "
                    f"still references it"
                )
                continue
            signals.pop(signal, None)

    write_yaml(path, config)
    return kept_tokens


def remove_python_classes(path: Path, class_names: Set[str]) -> None:
    if not path.exists() or not class_names:
        return

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: List[str] = []
    index = 0
    while index < len(lines):
        match = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\b", lines[index])
        if match and match.group(1) in class_names:
            index += 1
            while index < len(lines):
                if re.match(r"^(class|def)\s+[A-Za-z_][A-Za-z0-9_]*\b", lines[index]):
                    break
                index += 1
            continue
        output.append(lines[index])
        index += 1

    path.write_text("".join(output), encoding="utf-8")


def bracket_delta(line: str) -> int:
    return line.count("(") + line.count("[") + line.count("{") - line.count(")") - line.count("]") - line.count("}")


def remove_self_model_initializers(path: Path, model_keys: Set[str]) -> None:
    if not path.exists() or not model_keys:
        return

    assignment_re = re.compile(r"^(\s*)self\.([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: List[str] = []
    index = 0

    while index < len(lines):
        match = assignment_re.match(lines[index])
        if not match or match.group(2) not in model_keys:
            output.append(lines[index])
            index += 1
            continue

        depth = bracket_delta(lines[index])
        index += 1
        while index < len(lines) and depth > 0:
            depth += bracket_delta(lines[index])
            index += 1

    path.write_text("".join(output), encoding="utf-8")


def remove_files(root: Path, files: Iterable[Path]) -> None:
    for relative_path in files:
        path = root / relative_path
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def sanitize_residual_text(root: Path, tokens: Set[str]) -> None:
    text_suffixes = {".md", ".rst", ".txt"}
    code_suffixes = {".py", ".cpp", ".hpp", ".h", ".yaml", ".yml", ".xml"}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in {".git", "build", "install", "log", "dist"} for part in path.parts):
            continue
        if path.suffix not in text_suffixes and path.suffix not in code_suffixes:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            continue

        changed = False
        output: List[str] = []
        for line in lines:
            has_token = any(token and token in line for token in tokens)
            stripped = line.lstrip()
            if path.suffix in text_suffixes and has_token:
                changed = True
                continue
            if path.suffix in code_suffixes and has_token and stripped.startswith("#"):
                changed = True
                continue
            output.append(line)
        if changed:
            path.write_text("".join(output), encoding="utf-8")


def run_self_check(root: Path, tokens: Set[str]) -> None:
    leaks: List[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in {".git", "build", "install", "log", "dist"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in sorted(tokens):
            if token and token in text:
                leaks.append(f"{path.relative_to(root)}: {token}")
                break

    if leaks:
        preview = "\n".join(leaks[:50])
        raise RuntimeError("protected release strings remain:\n" + preview)


def removed_values(
    state_configs: Dict[str, ProtectedState],
    removed_states: Set[str],
    field_name: str,
) -> Set[str]:
    removed_configs = [config for state, config in state_configs.items() if state in removed_states]
    kept_configs = [config for state, config in state_configs.items() if state not in removed_states]
    return all_values(removed_configs, field_name) - all_values(kept_configs, field_name)


def removed_files(state_configs: Dict[str, ProtectedState], removed_states: Set[str]) -> Set[Path]:
    removed_configs = [config for state, config in state_configs.items() if state in removed_states]
    kept_configs = [config for state, config in state_configs.items() if state not in removed_states]
    return all_files(removed_configs) - all_files(kept_configs)


def sanitize_release(
    source_root: Path,
    output_root: Path,
    manifest_paths: List[Path],
    remote_config: Path,
    self_check: bool,
) -> None:
    specs = [load_protection_spec(source_root, path) for path in manifest_paths]

    copy_release_tree(source_root, output_root)

    protected_outputs: Set[str] = set()
    tokens_to_check: Set[str] = set()
    files_to_remove: Set[Path] = set(PUBLIC_DEV_ONLY_PATHS)
    for spec in specs:
        files_to_remove.add(spec.manifest_path)

        state_machine_path = output_root / spec.state_machine_config
        state_names = set(spec.states.keys())
        state_result = sanitize_state_machine(state_machine_path, state_names)

        behaviors = removed_values(spec.states, state_result.removed_states, "behaviors")
        model_keys = removed_values(spec.states, state_result.removed_states, "model_keys")
        explicit_files = removed_files(spec.states, state_result.removed_states)

        remove_python_classes(output_root / spec.robot_states, behaviors)
        remove_self_model_initializers(output_root / spec.demo_node, model_keys)

        files_to_remove.update(explicit_files)
        protected_outputs.update(state_result.outputs_to_remove)
        tokens_to_check.update(state_result.removed_states)
        tokens_to_check.update(behaviors)
        tokens_to_check.update(state_result.removed_events)
        tokens_to_check.update(model_keys)

    kept_remote_tokens = sanitize_remote_config(output_root / remote_config, tokens_to_check, protected_outputs)
    tokens_to_check -= kept_remote_tokens

    remove_files(output_root, files_to_remove)
    sanitize_residual_text(output_root, tokens_to_check)

    if self_check:
        run_self_check(output_root, tokens_to_check)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a public release tree without protected states.")
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="release protection manifest; can be passed more than once",
    )
    parser.add_argument(
        "--remote-config",
        default=str(DEFAULT_REMOTE_CONFIG),
        help="remote controller config to sanitize",
    )
    parser.add_argument("--out", default="dist/public_release", help="output directory")
    parser.add_argument("--self-check", action="store_true", help="fail if removed protected strings remain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path.cwd()
    manifest_paths = [Path(item) for item in args.manifest] if args.manifest else list(DEFAULT_MANIFESTS)
    output_root = Path(args.out)
    sanitize_release(source_root, output_root, manifest_paths, Path(args.remote_config), args.self_check)
    print(f"public release tree generated at: {output_root}")


if __name__ == "__main__":
    main()
