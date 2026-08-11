#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "remote_controller/config.hpp"
#include "remote_controller/input_mapper.hpp"

namespace remote_controller {

using DriverOutputHandler = std::function<void(const std::vector<std::string> &)>;
using DriverLogHandler = std::function<void(const std::string &)>;

class InputDriver {
public:
    virtual ~InputDriver() = default;

    virtual std::string name() const = 0;
    // Must be non-blocking.  Protocol-aware drivers may keep their own
    // lightweight probe state, but the manager calls this on its scan period.
    virtual bool is_available() = 0;
    // A driver becomes ready only after it has received the snapshot it says
    // is sufficient for safe input activation.
    virtual bool is_ready() const = 0;
    // Most drivers report immediate physical availability and let
    // InputDeviceManager apply loss_timeout_ms.  Protocol drivers that use
    // loss_timeout_ms inside is_available() (for example CRSF valid-frame
    // freshness) return true so the manager does not apply the timeout twice.
    virtual bool availability_handles_loss_timeout() const
    {
        return false;
    }
    // Emits driver-specific diagnostics.  InputDeviceManager invokes it only
    // when its runtime --DEBUG switch is enabled.
    virtual void debug() const
    {
    }
    virtual void start() = 0;
    virtual void stop() = 0;
};

// Convenience base for compiled-in drivers.  It centralizes synchronized
// signal delivery to InputMapper while leaving protocol parsing, availability
// probing, and readiness semantics to the concrete driver.
class InputDriverBase : public InputDriver {
public:
    InputDriverBase(
        InputDeviceConfig config,
        InputMapper &mapper,
        std::mutex &mapper_lock,
        DriverOutputHandler output_handler,
        DriverLogHandler log_handler);

protected:
    InputDeviceConfig config_;
    InputMapper &mapper_;
    std::mutex &mapper_lock_;
    DriverOutputHandler output_handler_;
    DriverLogHandler log_handler_;
    std::atomic<bool> stop_flag_{false};

    void log(const std::string &message) const;
    void dispatch(const std::vector<std::string> &outputs) const;
    void set_signal(const std::string &source, double value);
    void set_signals(const std::vector<std::pair<std::string, double>> &signals);
};

}  // namespace remote_controller
