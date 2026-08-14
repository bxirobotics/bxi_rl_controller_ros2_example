# SONIC vendor runtimes

This directory follows the framework's target-scoped vendor layout. SONIC
first probes every user-managed Python environment without these paths. Only
when none is usable does the launcher add the directory matching the current
OS, CPU architecture and Python ABI to the manager process.

```text
python/linux-x86_64-cpython-310/
python/linux-aarch64-cpython-310/
lib/linux-x86_64/
lib/linux-aarch64/
licenses/
```

## xrobotoolkit_sdk

The Python directories contain the installed binary extension from
`xrobotoolkit_sdk==1.0.2`, built from revision
`75cb1130ac63e76d8f7e7788049be415e8be44f2` of:

<https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git>

The extensions target CPython 3.10. Their SHA-256 values are:

- Linux x86_64: `5302ba646add3e44502c47681e0c5838e779615a074da87e2e4702718925fb7a`
- Linux aarch64: `a8532b442ba19c8e6bac75755f32f627c0ec7d6625b9c17610d7ded3017b0b2f`

Absolute build-machine `RUNPATH` entries were removed from both installed
extensions. Runtime lookup is supplied by the selected RoboticsService root. A
user installation is preferred; `vendor/lib/<platform>` exposes the matching
bundled SDK only as a portable fallback.

Each extension dynamically loads `libPXREARobotSDK.so`. The entries under
`lib/<platform>/` are relative symbolic links to the matching complete service
runtime under `../runtime/<platform>/roboticsservice/`; they expose that SDK
library to deployment checks and fallback tooling without duplicating the
large binary. They are not injected when a compatible user environment is
selected.

## RoboticsService

The complete service application and its private Qt/ICU closure remain under
`runtime/<platform>/roboticsservice/`, because they form one independently
launched application runtime rather than general Mod libraries. Each runtime
retains its own license, notice, provenance and `setting.ini`.

Canonical source notices used by `deploy_dependencies.sh` are stored under
`licenses/`; the BXI service configuration template is stored separately under
`../config/roboticsservice-minimal.ini`.

## MediaMTX

`runtime/linux-x86_64/mediamtx` and `runtime/linux-aarch64/mediamtx` are the
unmodified official MediaMTX 1.15.6 static executables used by SONIC's
state-scoped RTSP server. Their upstream release URLs, archive hashes and
binary hashes are recorded in `licenses/MediaMTX.PROVENANCE.txt`; the MIT
license is retained in `licenses/MediaMTX.LICENSE`.
