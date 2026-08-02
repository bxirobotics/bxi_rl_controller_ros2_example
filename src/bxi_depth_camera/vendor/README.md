# Depth camera vendor runtimes

This directory contains the prebuilt fallback RealSense and Orbbec Python SDK
runtimes used only by the standalone `bxi_depth_camera` process.

The launcher probes the host environment first. A matching directory below is
added only when the corresponding system SDK cannot be imported, so these
artifacts never replace a working user installation and never enter the
controller process' `PYTHONPATH`.

- Target: Linux x86_64, CPython 3.10 (`cp310`)
- `python/linux-x86_64-cpython-310/pyrealsense2`: PyPI
  `pyrealsense2==2.58.1.10581`
- `lib/linux-x86_64/librealsense2.so.2.57.7`: ROS Humble package
  `ros-humble-librealsense2==2.57.7-1jammy.20260324.115117`

SHA-256:

- `pyrealsense2.cpython-310-x86_64-linux-gnu.so`:
  `1e075c5740bb685b39396289cb636b6679c2ce4b0c7d95c5077dc545c79882a0`
- `librealsense2.so.2.57.7`:
  `c93d52c5ab2d91d79e2b591775b739d94bc548f7d365c4fdc496f6a4db5f46f6`

The Python extension includes the RealSense SDK implementation and dynamically
links the target system's glibc, libstdc++, libusb and libudev. Those base OS
libraries, ROS 2 itself and `sensor_msgs` are intentionally not bundled here.

The runtime only adds the directory matching its OS, CPU architecture and
Python ABI. Cross-platform pure Python dependencies belong in `python/common`.

`licenses/` and the Python distribution metadata retain the upstream license
notices. Replace these artifacts as a unit when changing the target Python ABI,
CPU architecture or RealSense SDK version.

## Orbbec Python SDK runtime

The Gemini 335 node calls the official SDK directly through
`pyorbbecsdk2==2.1.1` (Orbbec SDK `2.8.6`). It does not use OrbbecSDK_ROS2.
The CPython 3.10 wheels are unpacked and pruned to the importable extension,
SDK shared library, configuration, filters, udev setup files and distribution
license metadata:

```text
python/linux-x86_64-cpython-310/pyorbbecsdk/
python/linux-aarch64-cpython-310/pyorbbecsdk/
```

Source wheel SHA-256:

- Linux x86_64 CPython 3.10:
  `1cbd4c630f7edd7588299d7231169508253780c5c0bb658970485cf62c14cfeb`
- Linux aarch64 CPython 3.10:
  `dfe360ca7c6e595bc0d55d215cebb897911fd4992ce5814fd8878cb91806c9ee`

Bundled binary SHA-256:

- x86_64 `pyorbbecsdk` extension:
  `38a87362644a34a73e64247624c7235e724c5c7f2b49ac6ca127bdf92e7a6232`
- x86_64 `libOrbbecSDK.so.2.8.6`:
  `96df94751c3fc7dc72a95b75d40a5a89a768a03767de8e197ab4cde9a8a89b28`
- aarch64 `pyorbbecsdk` extension:
  `54f91da5b4d1df2758297fcf6e7e013f9f22ef818d24e4c2146ac0ff69f8b420`
- aarch64 `libOrbbecSDK.so.2.8.6`:
  `ff9646864b793056d7f5137ff885a39c04307a2930edf28205cd2423b46ba1bc`

The three identical SDK library files from each wheel are stored as one
versioned file plus SONAME symlinks. Examples, GUI dependencies and the bundled
YOLO model are intentionally omitted. The Apache-2.0 notice remains under each
`pyorbbecsdk2-2.1.1.dist-info/licenses/` directory. Linux udev rules remain
under `pyorbbecsdk/shared/`. The bundled SDK configuration changes only
`FileLogLevel` from `INFO` to `OFF` so a Mod run does not create a `Log/`
directory in its working tree; warning-level console logging remains enabled.
