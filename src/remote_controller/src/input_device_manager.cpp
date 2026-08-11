#include "remote_controller/input_device_manager.hpp"

#include <algorithm>
#include <utility>

namespace remote_controller {

struct InputDeviceManager::Candidate {
    InputDeviceConfig config;
    std::unique_ptr<InputDriver> driver;
    bool available = false;
    std::chrono::steady_clock::time_point available_since{};
    std::chrono::steady_clock::time_point unavailable_since{};
    std::chrono::steady_clock::time_point cooldown_until{};
    std::chrono::steady_clock::time_point started_at{};
};

InputDeviceManager::InputDeviceManager(
    const RemoteConfig &config,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler,
    bool debug_enabled,
    std::string driver_filter)
    : mapper_(mapper),
      mapper_lock_(mapper_lock),
      output_handler_(std::move(output_handler)),
      log_handler_(std::move(log_handler)),
      debug_enabled_(debug_enabled),
      driver_filter_(std::move(driver_filter))
{
    {
        const std::lock_guard<std::mutex> guard(mapper_lock_);
        mapper_.set_debug_enabled(debug_enabled_);
    }
    std::lock_guard<std::mutex> guard(state_lock_);
    configure_locked(config);
}

InputDeviceManager::~InputDeviceManager()
{
    stop();
}

void InputDeviceManager::log(const std::string &message) const
{
    if (log_handler_) {
        log_handler_(message);
    }
}

void InputDeviceManager::configure_locked(const RemoteConfig &config)
{
    candidates_.clear();
    selection_ = config.input_selection;
    running_index_ = -1;
    running_ready_ = false;
    accepting_outputs_ = false;
    desired_index_ = -1;
    safe_output_pending_ = false;
    safe_output_published_ = false;
    next_scan_ = std::chrono::steady_clock::time_point{};
    next_debug_ = std::chrono::steady_clock::time_point{};

    for (const auto &device : config.input_devices) {
        if (!driver_filter_.empty() && device.type != driver_filter_) {
            continue;
        }
        if (!has_input_driver_factory(device.type)) {
            log("input device '" + device.name + "' ignored: driver type '" +
                device.type + "' is not compiled in");
            continue;
        }

        std::unique_ptr<Candidate> candidate(new Candidate());
        candidate->config = device;
        candidate->driver = create_input_driver(
            device,
            mapper_,
            mapper_lock_,
            [this, name = device.name](const std::vector<std::string> &outputs) {
                handle_driver_outputs(name, outputs);
            },
            [this, name = device.name](const std::string &message) {
                log("input " + name + ": " + message);
            });
        if (!candidate->driver) {
            log("input device '" + device.name + "' ignored: factory did not create a driver");
            continue;
        }
        candidates_.push_back(std::move(candidate));
    }

    if (candidates_.empty()) {
        log("no supported input devices are configured; controller will remain safe-stopped");
    }
}

void InputDeviceManager::stop()
{
    std::vector<InputDriver *> drivers;
    std::set<std::string> sources;
    {
        const std::lock_guard<std::mutex> guard(state_lock_);
        for (const auto &candidate : candidates_) {
            if (candidate->driver) {
                drivers.push_back(candidate->driver.get());
            }
            sources.insert(candidate->config.raw_sources.begin(), candidate->config.raw_sources.end());
        }
        running_index_ = -1;
        running_ready_ = false;
        accepting_outputs_ = false;
        desired_index_ = -1;
        safe_output_pending_ = false;
        safe_output_published_ = false;
    }

    for (InputDriver *driver : drivers) {
        driver->stop();
    }
    if (!sources.empty()) {
        const std::lock_guard<std::mutex> guard(mapper_lock_);
        mapper_.set_input_edges_enabled(false);
        mapper_.clear_signals(sources);
    }
}

void InputDeviceManager::update_availability_locked(std::chrono::steady_clock::time_point now)
{
    for (const auto &candidate : candidates_) {
        const bool available = candidate->driver->is_available();
        if (available && !candidate->available) {
            candidate->available = true;
            candidate->available_since = now;
            candidate->unavailable_since = std::chrono::steady_clock::time_point{};
            log("input candidate available: " + candidate->config.name);
        } else if (!available && candidate->available) {
            candidate->available = false;
            candidate->available_since = std::chrono::steady_clock::time_point{};
            candidate->unavailable_since = now;
            log("input candidate unavailable: " + candidate->config.name);
        }
    }
}

void InputDeviceManager::debug_drivers(std::chrono::steady_clock::time_point now)
{
    if (!debug_enabled_) {
        return;
    }

    std::vector<InputDriver *> drivers;
    {
        const std::lock_guard<std::mutex> guard(state_lock_);
        if (now < next_debug_) {
            return;
        }
        next_debug_ = now + std::chrono::seconds(1);
        for (const auto &candidate : candidates_) {
            if (candidate->driver) {
                drivers.push_back(candidate->driver.get());
            }
        }
    }

    for (const auto *driver : drivers) {
        driver->debug();
    }
    std::vector<std::string> control_messages;
    {
        const std::lock_guard<std::mutex> guard(mapper_lock_);
        control_messages = mapper_.take_debug_messages();
    }
    for (const auto &message : control_messages) {
        log(message);
    }
}

int InputDeviceManager::best_ready_candidate_locked(std::chrono::steady_clock::time_point now) const
{
    int best_index = -1;
    for (std::size_t index = 0; index < candidates_.size(); ++index) {
        const Candidate &candidate = *candidates_[index];
        if (!candidate.available || now < candidate.cooldown_until) {
            continue;
        }
        if (candidate.available_since == std::chrono::steady_clock::time_point{} ||
            now - candidate.available_since <
                std::chrono::milliseconds(selection_.promote_stable_ms)) {
            continue;
        }
        if (best_index < 0 ||
            candidate.config.priority > candidates_[best_index]->config.priority ||
            (candidate.config.priority == candidates_[best_index]->config.priority &&
             candidate.config.name < candidates_[best_index]->config.name)) {
            best_index = static_cast<int>(index);
        }
    }
    return best_index;
}

bool InputDeviceManager::require_safe_output(int desired_index)
{
    const std::lock_guard<std::mutex> guard(state_lock_);
    desired_index_ = desired_index;
    if (safe_output_pending_) {
        return false;
    }
    safe_output_pending_ = true;
    safe_output_published_ = false;
    log("input safety barrier: publishing one zero-motion command before switching");
    return true;
}

void InputDeviceManager::start_candidate(int index)
{
    InputDriver *driver = nullptr;
    std::set<std::string> sources;
    std::string name;
    {
        const std::lock_guard<std::mutex> guard(state_lock_);
        if (index < 0 || index >= static_cast<int>(candidates_.size()) ||
            running_index_ >= 0 || safe_output_pending_) {
            return;
        }
        Candidate &candidate = *candidates_[index];
        if (!candidate.available ||
            std::chrono::steady_clock::now() < candidate.cooldown_until) {
            return;
        }
        running_index_ = index;
        running_ready_ = false;
        accepting_outputs_ = false;
        candidate.started_at = std::chrono::steady_clock::now();
        driver = candidate.driver.get();
        sources = candidate.config.raw_sources;
        name = candidate.config.name;
    }

    {
        const std::lock_guard<std::mutex> guard(mapper_lock_);
        mapper_.set_input_edges_enabled(false);
        mapper_.clear_signals(sources);
    }

    try {
        driver->start();
        log("starting input candidate: " + name);
    } catch (const std::exception &exception) {
        log("input candidate " + name + " failed to start: " + exception.what());
        stop_running("start failure", true);
    }
}

void InputDeviceManager::stop_running(const std::string &reason, bool cooldown)
{
    InputDriver *driver = nullptr;
    std::set<std::string> sources;
    std::string name;
    {
        const std::lock_guard<std::mutex> guard(state_lock_);
        if (running_index_ < 0 || running_index_ >= static_cast<int>(candidates_.size())) {
            return;
        }
        Candidate &candidate = *candidates_[running_index_];
        driver = candidate.driver.get();
        sources = candidate.config.raw_sources;
        name = candidate.config.name;
        if (cooldown) {
            candidate.cooldown_until = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(candidate.config.cooldown_ms);
        }
        running_index_ = -1;
        running_ready_ = false;
        accepting_outputs_ = false;
    }

    driver->stop();
    {
        const std::lock_guard<std::mutex> guard(mapper_lock_);
        mapper_.set_input_edges_enabled(false);
        mapper_.clear_signals(sources);
    }
    log("input candidate stopped: " + name + " (" + reason + ")");
}

bool InputDeviceManager::tick()
{
    const auto now = std::chrono::steady_clock::now();
    bool scan = false;
    {
        const std::lock_guard<std::mutex> guard(state_lock_);
        if (now >= next_scan_) {
            next_scan_ = now + std::chrono::milliseconds(selection_.scan_interval_ms);
            scan = true;
            update_availability_locked(now);
        }
    }
    debug_drivers(now);

    bool stop_for_loss = false;
    bool stop_for_timeout = false;
    bool promote = false;
    int desired = -1;
    bool became_ready = false;
    {
        const std::lock_guard<std::mutex> guard(state_lock_);
        if (running_index_ >= 0) {
            Candidate &running = *candidates_[running_index_];
            if (!running.available) {
                if (running.unavailable_since == std::chrono::steady_clock::time_point{}) {
                    running.unavailable_since = now;
                }
                if (running.driver->availability_handles_loss_timeout() ||
                    now - running.unavailable_since >=
                        std::chrono::milliseconds(running.config.loss_timeout_ms)) {
                    stop_for_loss = true;
                }
            } else {
                running.unavailable_since = std::chrono::steady_clock::time_point{};
            }

            if (!running_ready_) {
                if (running.driver->is_ready()) {
                    running_ready_ = true;
                    accepting_outputs_ = true;
                    safe_output_published_ = false;
                    became_ready = true;
                } else if (now - running.started_at >=
                    std::chrono::milliseconds(running.config.ready_timeout_ms)) {
                    stop_for_timeout = true;
                }
            }

            if (running_ready_ && !stop_for_loss) {
                const int best = best_ready_candidate_locked(now);
                if (best >= 0 && candidates_[best]->config.priority > running.config.priority) {
                    desired = best;
                    promote = true;
                }
            }
        } else {
            desired = best_ready_candidate_locked(now);
        }
    }

    if (became_ready) {
        std::string ready_name;
        {
            const std::lock_guard<std::mutex> guard(state_lock_);
            if (running_index_ >= 0) {
                ready_name = candidates_[running_index_]->config.name;
            }
        }
        const std::lock_guard<std::mutex> guard(mapper_lock_);
        mapper_.set_input_edges_enabled(true);
        log("input candidate ready; accepting commands: " + ready_name);
    }

    if (stop_for_loss || stop_for_timeout || promote) {
        stop_running(
            promote ? "higher-priority candidate available" :
            (stop_for_loss ? "availability lost" : "ready timeout"),
            !promote);
        int fallback = desired;
        if (fallback < 0) {
            const std::lock_guard<std::mutex> guard(state_lock_);
            fallback = best_ready_candidate_locked(now);
        }
        return require_safe_output(fallback);
    }

    if (!scan) {
        return false;
    }

    bool has_running = false;
    bool safe_published = false;
    bool safe_pending = false;
    {
        const std::lock_guard<std::mutex> guard(state_lock_);
        has_running = running_index_ >= 0;
        safe_published = safe_output_published_;
        safe_pending = safe_output_pending_;
        if (!has_running) {
            desired = best_ready_candidate_locked(now);
        }
    }
    if (has_running || safe_pending) {
        return false;
    }
    if (!safe_published) {
        return require_safe_output(desired);
    }
    if (desired >= 0) {
        start_candidate(desired);
    }
    return false;
}

void InputDeviceManager::notify_safe_output_published()
{
    int desired = -1;
    {
        const std::lock_guard<std::mutex> guard(state_lock_);
        if (!safe_output_pending_) {
            return;
        }
        safe_output_pending_ = false;
        safe_output_published_ = true;
        desired = desired_index_;
        desired_index_ = -1;
    }
    if (desired >= 0) {
        start_candidate(desired);
    }
}

void InputDeviceManager::handle_driver_outputs(
    const std::string &name,
    const std::vector<std::string> &outputs)
{
    bool accept = false;
    {
        const std::lock_guard<std::mutex> guard(state_lock_);
        if (running_index_ >= 0 && running_ready_ && accepting_outputs_ &&
            candidates_[running_index_]->config.name == name) {
            accept = true;
        }
    }
    if (accept && output_handler_) {
        output_handler_(outputs);
    }
}

}  // namespace remote_controller
