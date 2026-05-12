#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "remote_controller/input_mapper.hpp"

namespace remote_controller {

using DriverOutputHandler = std::function<void(const std::vector<std::string> &)>;
using DriverLogHandler = std::function<void(const std::string &)>;

class InputDriver {
public:
    virtual ~InputDriver() = default;

    virtual std::string name() const = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
};

std::unique_ptr<InputDriver> create_input_driver(
    const std::string &driver_type,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler);

}  // namespace remote_controller
