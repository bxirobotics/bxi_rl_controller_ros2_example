#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "remote_controller/config.hpp"
#include "remote_controller/drivers/driver_registry.hpp"

namespace remote_controller {

// Selects exactly one input device according to YAML priority.  The manager
// owns lifecycle transitions so a disconnect can never leave old input values
// active while a fallback device is being brought online.
class InputDeviceManager {
public:
    InputDeviceManager(
        const RemoteConfig &config,
        InputMapper &mapper,
        std::mutex &mapper_lock,
        DriverOutputHandler output_handler,
        DriverLogHandler log_handler,
        bool debug_enabled = false,
        std::string driver_filter = "");
    ~InputDeviceManager();

    InputDeviceManager(const InputDeviceManager &) = delete;
    InputDeviceManager &operator=(const InputDeviceManager &) = delete;

    void stop();

    // Returns true when the caller must publish one all-zero command before
    // notify_safe_output_published() may start the next candidate.
    bool tick();
    void notify_safe_output_published();

private:
    struct Candidate;

    InputMapper &mapper_;
    std::mutex &mapper_lock_;
    DriverOutputHandler output_handler_;
    DriverLogHandler log_handler_;
    bool debug_enabled_ = false;
    std::string driver_filter_;
    InputSelectionConfig selection_;
    std::vector<std::unique_ptr<Candidate>> candidates_;
    std::mutex state_lock_;
    int running_index_ = -1;
    bool running_ready_ = false;
    bool accepting_outputs_ = false;
    int desired_index_ = -1;
    bool safe_output_pending_ = false;
    bool safe_output_published_ = false;
    std::chrono::steady_clock::time_point next_scan_{};
    std::chrono::steady_clock::time_point next_debug_{};

    void log(const std::string &message) const;
    void configure_locked(const RemoteConfig &config);
    void debug_drivers(std::chrono::steady_clock::time_point now);
    int best_ready_candidate_locked(std::chrono::steady_clock::time_point now) const;
    void update_availability_locked(std::chrono::steady_clock::time_point now);
    void start_candidate(int index);
    void stop_running(const std::string &reason, bool cooldown);
    bool require_safe_output(int desired_index);
    void handle_driver_outputs(const std::string &name, const std::vector<std::string> &outputs);
};

}  // namespace remote_controller
