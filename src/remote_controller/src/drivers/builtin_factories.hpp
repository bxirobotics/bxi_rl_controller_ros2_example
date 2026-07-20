#pragma once

#include "remote_controller/drivers/input_driver.hpp"

namespace remote_controller {

std::unique_ptr<InputDriver> create_joystick_input_driver(
    const InputDeviceConfig &config,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler);

std::unique_ptr<InputDriver> create_keyboard_input_driver(
    const InputDeviceConfig &config,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler);

std::unique_ptr<InputDriver> create_crsf_input_driver(
    const InputDeviceConfig &config,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler);

}  // namespace remote_controller
