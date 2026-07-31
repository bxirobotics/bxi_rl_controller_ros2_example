"""Child-process runner for a Python node declared by a Mod."""

from __future__ import annotations

import argparse
from pathlib import Path

from bxi_example_py_elf3.framework.mod_api.node import NodeBuildContext


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    from bxi_example_py_elf3.framework.runtime.mod_loader import load_process_node_spec

    spec, _module_prefix = load_process_node_spec(args.manifest, args.node)
    context = NodeBuildContext(
        mod_id=spec.mod_id,
        node_id=spec.id,
        node_name=spec.node_name,
        mod_root=spec.mod_root,
        params=spec.params,
        arguments=spec.arguments,
        remappings=spec.remappings,
        namespace=spec.namespace,
    )

    rclpy.init(args=[])
    node = None
    executor = SingleThreadedExecutor()
    try:
        factory = spec.factory
        if factory is None:
            raise RuntimeError(f"Mod node '{spec.id}' has no process factory")
        node = factory(context)
        if not callable(getattr(node, "destroy_node", None)):
            raise TypeError(
                f"Mod node entrypoint '{spec.entrypoint}' must return an rclpy Node"
            )
        if executor.add_node(node) is False:
            raise RuntimeError(f"executor rejected Mod node '{spec.id}'")
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
    finally:
        try:
            executor.shutdown()
        finally:
            try:
                if node is not None:
                    node.destroy_node()
            finally:
                if rclpy.ok():
                    rclpy.shutdown()


if __name__ == "__main__":
    main()
