#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "remote_controller/drivers/input_driver.hpp"

namespace remote_controller {

using InputDriverFactory = std::function<std::unique_ptr<InputDriver>(
    const InputDeviceConfig &,
    InputMapper &,
    std::mutex &,
    DriverOutputHandler,
    DriverLogHandler)>;

// New compiled-in drivers register a factory during application startup.  An
// unregistered type is intentionally non-fatal: the device manager logs it
// and continues to lower-priority, supported candidates.
bool register_input_driver_factory(const std::string &type, InputDriverFactory factory);
bool has_input_driver_factory(const std::string &type);

std::unique_ptr<InputDriver> create_input_driver(
    const InputDeviceConfig &config,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler);

}  // namespace remote_controller
