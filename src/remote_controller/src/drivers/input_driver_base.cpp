#include "remote_controller/drivers/input_driver.hpp"

#include <utility>

namespace remote_controller {

InputDriverBase::InputDriverBase(
    InputDeviceConfig config,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler)
    : config_(std::move(config)),
      mapper_(mapper),
      mapper_lock_(mapper_lock),
      output_handler_(std::move(output_handler)),
      log_handler_(std::move(log_handler))
{
}

void InputDriverBase::log(const std::string &message) const
{
    if (log_handler_) {
        log_handler_(message);
    }
}

void InputDriverBase::dispatch(const std::vector<std::string> &outputs) const
{
    if (!outputs.empty() && output_handler_) {
        output_handler_(outputs);
    }
}

void InputDriverBase::set_signal(const std::string &source, double value)
{
    std::vector<std::string> outputs;
    {
        const std::lock_guard<std::mutex> guard(mapper_lock_);
        outputs = mapper_.set_signal(source, value);
    }
    dispatch(outputs);
}

void InputDriverBase::set_signals(const std::vector<std::pair<std::string, double>> &signals)
{
    if (signals.empty()) {
        return;
    }
    std::vector<std::string> outputs;
    {
        const std::lock_guard<std::mutex> guard(mapper_lock_);
        outputs = mapper_.set_signals(signals);
    }
    dispatch(outputs);
}

}  // namespace remote_controller
