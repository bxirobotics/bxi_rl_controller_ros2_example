# bxi_depth_camera SDK bundle

- Platform: `linux-x86_64`
- Baseline: Ubuntu 22.04 / glibc 2.35
- Orbbec SDK: `OrbbecSDK_v2 v2.9.3`
- Orbbec source: `https://github.com/orbbec/OrbbecSDK_v2.git`
- Orbbec source commit: `2f6561c28255d805b34aa00a690199ce40e96c81`
- Orbbec library SHA-256: `f1c16d63afb1eedf103f96bc382c22625dd8d3975e51218d41888a9b81342fbc`
- librealsense: `v2.57.7`
- librealsense source: `https://github.com/IntelRealSense/librealsense.git`
- librealsense library SHA-256: `c93d52c5ab2d91d79e2b591775b739d94bc548f7d365c4fdc496f6a4db5f46f6`

The Orbbec core was built from the pinned source above. Its matching headers,
runtime XML, extension libraries, udev rule and licenses are included. The
librealsense library and headers are from the ROS 2 Humble `2.57.7` binary
distribution; its udev rules and Apache-2.0 license are included.

The bundle deliberately does not include glibc, libstdc++, libudev or libusb.
Those are platform libraries and must come from the target operating system.
