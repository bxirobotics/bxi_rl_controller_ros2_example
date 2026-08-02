from __future__ import annotations

import os
import sys

from .vendor import bootstrap_environment


def main() -> None:
    environment = bootstrap_environment(("pyrealsense2", "pyorbbecsdk"))
    if environment is not None:
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "bxi_depth_camera.manager", *sys.argv[1:]],
            environment,
        )
    from .manager import main as manager_main

    manager_main()


if __name__ == "__main__":
    main()
