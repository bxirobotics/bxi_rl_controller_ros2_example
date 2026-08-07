cmake_minimum_required(VERSION 3.14)
project(bxi_depth_camera VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ROS 2 Humble is built against the Ubuntu C++ ABI. An activated Conda
# environment otherwise becomes an implicit CMake prefix and can inject its
# incompatible fmt/TBB libraries into the link graph.
if(DEFINED ENV{CONDA_PREFIX} AND IS_DIRECTORY "$ENV{CONDA_PREFIX}")
  list(APPEND CMAKE_IGNORE_PATH
    "$ENV{CONDA_PREFIX}/bin"
    "$ENV{CONDA_PREFIX}/include"
    "$ENV{CONDA_PREFIX}/lib"
    "$ENV{CONDA_PREFIX}/lib/cmake"
    "$ENV{CONDA_PREFIX}/lib/cmake/fmt"
    "$ENV{CONDA_PREFIX}/lib/cmake/TBB"
  )
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" BXI_SYSTEM_PROCESSOR)
if(BXI_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(BXI_VENDOR_PLATFORM "linux-x86_64")
elseif(BXI_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(BXI_VENDOR_PLATFORM "linux-aarch64")
else()
  message(FATAL_ERROR
    "bxi_depth_camera has no SDK bundle for CMAKE_SYSTEM_PROCESSOR="
    "${CMAKE_SYSTEM_PROCESSOR}. Supported platforms: x86_64 and aarch64.")
endif()

set(BXI_VENDOR_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp/${BXI_VENDOR_PLATFORM}")
set(BXI_ORBBEC_ROOT "${BXI_VENDOR_ROOT}/orbbec_sdk_v2")
set(BXI_REALSENSE_ROOT "${BXI_VENDOR_ROOT}/realsense2")
set(BXI_ORBBEC_LIBRARY
  "${BXI_ORBBEC_ROOT}/lib/libOrbbecSDK.so.2.9.3")
set(BXI_REALSENSE_LIBRARY
  "${BXI_REALSENSE_ROOT}/lib/librealsense2.so.2.57.7")

set(BXI_REQUIRED_VENDOR_FILES
  "${BXI_ORBBEC_ROOT}/include/libobsensor/ObSensor.hpp"
  "${BXI_ORBBEC_LIBRARY}"
  "${BXI_ORBBEC_ROOT}/lib/OrbbecSDKConfig.xml"
  "${BXI_REALSENSE_ROOT}/include/librealsense2/rs.hpp"
  "${BXI_REALSENSE_LIBRARY}"
)
foreach(BXI_REQUIRED_FILE IN LISTS BXI_REQUIRED_VENDOR_FILES)
  if(NOT EXISTS "${BXI_REQUIRED_FILE}")
    message(FATAL_ERROR
      "Incomplete ${BXI_VENDOR_PLATFORM} camera SDK bundle: missing "
      "${BXI_REQUIRED_FILE}. Generate it with tools/build_sdk_bundle.sh on "
      "the target architecture and copy the resulting directory into "
      "vendor/cpp/.")
  endif()
endforeach()

add_library(bxi_vendor_orbbec SHARED IMPORTED GLOBAL)
set_target_properties(bxi_vendor_orbbec PROPERTIES
  IMPORTED_LOCATION "${BXI_ORBBEC_LIBRARY}"
  IMPORTED_SONAME "libOrbbecSDK.so.2"
  INTERFACE_INCLUDE_DIRECTORIES "${BXI_ORBBEC_ROOT}/include"
)
add_library(ob::OrbbecSDK ALIAS bxi_vendor_orbbec)

add_library(bxi_vendor_realsense SHARED IMPORTED GLOBAL)
set_target_properties(bxi_vendor_realsense PROPERTIES
  IMPORTED_LOCATION "${BXI_REALSENSE_LIBRARY}"
  IMPORTED_SONAME "librealsense2.so.2.57"
  INTERFACE_INCLUDE_DIRECTORIES "${BXI_REALSENSE_ROOT}/include"
)
add_library(realsense2::realsense2 ALIAS bxi_vendor_realsense)

add_library(bxi_depth_camera_core SHARED
  src/common.cpp
  src/manager.cpp
  src/orbbec_camera.cpp
  src/realsense_camera.cpp
)
target_include_directories(bxi_depth_camera_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(bxi_depth_camera_core
  ob::OrbbecSDK
  realsense2::realsense2
  ${OpenCV_LIBRARIES}
)
target_include_directories(bxi_depth_camera_core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(bxi_depth_camera_core
  rclcpp
  rcl_interfaces
  sensor_msgs
)
target_compile_options(bxi_depth_camera_core PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(bxi_depth_camera_core PROPERTIES
  BUILD_RPATH "${BXI_ORBBEC_ROOT}/lib;${BXI_REALSENSE_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN"
)

add_executable(cameras src/main.cpp)
target_link_libraries(cameras bxi_depth_camera_core)
set_target_properties(cameras PROPERTIES INSTALL_RPATH "$ORIGIN/..")

add_executable(cameras-inspect src/inspect.cpp)
target_link_libraries(cameras-inspect
  ob::OrbbecSDK
  realsense2::realsense2
)
target_compile_definitions(cameras-inspect PRIVATE
  BXI_DEPTH_CAMERA_TOOL_MODE=0
)
target_compile_options(cameras-inspect PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(cameras-inspect PROPERTIES
  BUILD_RPATH "${BXI_ORBBEC_ROOT}/lib;${BXI_REALSENSE_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN/.."
)

add_executable(cameras-validate src/inspect.cpp)
target_link_libraries(cameras-validate
  ob::OrbbecSDK
  realsense2::realsense2
)
target_compile_definitions(cameras-validate PRIVATE
  BXI_DEPTH_CAMERA_TOOL_MODE=1
)
target_compile_options(cameras-validate PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(cameras-validate PROPERTIES
  BUILD_RPATH "${BXI_ORBBEC_ROOT}/lib;${BXI_REALSENSE_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN/.."
)

add_executable(cameras-fps-test src/inspect.cpp)
target_link_libraries(cameras-fps-test
  ob::OrbbecSDK
  realsense2::realsense2
)
target_compile_definitions(cameras-fps-test PRIVATE
  BXI_DEPTH_CAMERA_TOOL_MODE=2
)
target_compile_options(cameras-fps-test PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(cameras-fps-test PROPERTIES
  BUILD_RPATH "${BXI_ORBBEC_ROOT}/lib;${BXI_REALSENSE_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN/.."
)

install(
  TARGETS bxi_depth_camera_core cameras cameras-inspect cameras-validate
          cameras-fps-test
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES launch/cameras.launch.py DESTINATION share/${PROJECT_NAME}/launch)
install(FILES README.md DESTINATION share/${PROJECT_NAME})
install(
  PROGRAMS tools/install_udev_rules.sh
  DESTINATION lib/${PROJECT_NAME}
  RENAME install-udev-rules
)
install(
  DIRECTORY "${BXI_ORBBEC_ROOT}/lib/"
  DESTINATION lib
  USE_SOURCE_PERMISSIONS
)
install(
  DIRECTORY "${BXI_REALSENSE_ROOT}/lib/"
  DESTINATION lib
  USE_SOURCE_PERMISSIONS
)
install(
  DIRECTORY "${BXI_ORBBEC_ROOT}/shared/"
  DESTINATION share/${PROJECT_NAME}/vendor/orbbec_sdk_v2
  USE_SOURCE_PERMISSIONS
)
install(
  DIRECTORY "${BXI_REALSENSE_ROOT}/shared/"
  DESTINATION share/${PROJECT_NAME}/vendor/realsense2
  USE_SOURCE_PERMISSIONS
)
install(
  DIRECTORY "${BXI_ORBBEC_ROOT}/licenses/"
  DESTINATION share/${PROJECT_NAME}/vendor/licenses/orbbec_sdk_v2
)
install(
  DIRECTORY "${BXI_REALSENSE_ROOT}/licenses/"
  DESTINATION share/${PROJECT_NAME}/vendor/licenses/realsense2
)
install(
  FILES "${BXI_VENDOR_ROOT}/MANIFEST.md"
  DESTINATION share/${PROJECT_NAME}/vendor
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_common test/test_common.cpp)
  if(TARGET test_common)
    target_link_libraries(test_common bxi_depth_camera_core)
  endif()
endif()

ament_package()
