# bxi_depth_camera vendor dependencies

`cpp/<platform>/` contains the native SDK bundle selected by CMake for the
current CPU architecture. A complete platform directory contains:

- OrbbecSDK_v2 `2.9.3` headers, shared library, extension libraries, runtime
  configuration, udev rule and licenses;
- librealsense `2.57.7` headers, shared library, udev rules and license;
- `MANIFEST.md` with exact provenance and checksums.

Supported directory names are:

```text
cpp/linux-x86_64
cpp/linux-aarch64
```

The package build never downloads or compiles either camera SDK and does not search for
a system SDK. It imports the matching bundle and installs its runtime libraries into the
ROS install space. glibc, libstdc++, libudev and libusb remain operating-system
dependencies.

To create a missing platform bundle natively, run `tools/build_sdk_bundle.sh` from the
package. Bundle generation is a maintainer operation, not part of normal package builds.
There are no Python SDK runtimes.
