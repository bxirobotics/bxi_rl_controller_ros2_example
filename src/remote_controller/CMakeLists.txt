cmake_minimum_required(VERSION 3.5)
project(remote_controller)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(communication REQUIRED)
find_package(yaml-cpp REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  add_executable(${PROJECT_NAME}_crsf_frame_parser_test
    test/crsf_frame_parser_test.cpp
  )
  target_include_directories(${PROJECT_NAME}_crsf_frame_parser_test PRIVATE include)
  add_test(
    NAME ${PROJECT_NAME}_crsf_frame_parser_test
    COMMAND ${PROJECT_NAME}_crsf_frame_parser_test
  )
endif()

add_library(${PROJECT_NAME}_core
  src/config.cpp
  src/drivers/driver_registry.cpp
  src/drivers/crsf_input_driver.cpp
  src/drivers/input_driver_base.cpp
  src/drivers/joystick_input_driver.cpp
  src/drivers/keyboard_input_driver.cpp
  src/input_device_manager.cpp
  src/input_mapper.cpp
  src/motion_commands_adapter.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME}_core communication)
target_link_libraries(${PROJECT_NAME}_core yaml-cpp)

if(BUILD_TESTING)
  add_executable(${PROJECT_NAME}_control_rules_test
    test/control_rules_test.cpp
  )
  target_include_directories(${PROJECT_NAME}_control_rules_test PRIVATE include)
  target_compile_definitions(${PROJECT_NAME}_control_rules_test PRIVATE
    REMOTE_CONTROLLER_TEST_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/xbox_default.yaml"
  )
  ament_target_dependencies(${PROJECT_NAME}_control_rules_test communication)
  target_link_libraries(${PROJECT_NAME}_control_rules_test ${PROJECT_NAME}_core)
  add_test(
    NAME ${PROJECT_NAME}_control_rules_test
    COMMAND ${PROJECT_NAME}_control_rules_test
  )
endif()

add_executable(${PROJECT_NAME}
  src/main.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE include)
ament_target_dependencies(${PROJECT_NAME} rclcpp std_msgs communication)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_core)

install(TARGETS ${PROJECT_NAME}_core ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
        DESTINATION include)

install(DIRECTORY launch
         DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config
         DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_core)
ament_export_targets(${PROJECT_NAME}_targets)
ament_export_dependencies(communication yaml-cpp)

ament_package()
