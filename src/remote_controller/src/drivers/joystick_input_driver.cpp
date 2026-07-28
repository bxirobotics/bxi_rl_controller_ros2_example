#include "remote_controller/drivers/input_driver.hpp"

#include <cerrno>
#include <fcntl.h>
#include <linux/joystick.h>
#include <set>
#include <string>
#include <sys/select.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace remote_controller {
namespace {

class JoystickInputDriver : public InputDriverBase {
public:
    using InputDriverBase::InputDriverBase;

    ~JoystickInputDriver() override
    {
        stop();
    }

    std::string name() const override
    {
        return config_.name;
    }

    bool is_available() override
    {
        if (available_) {
            return true;
        }
        if (config_.device.empty()) {
            return false;
        }
        const int probe_fd = open(config_.device.c_str(), O_RDONLY | O_NONBLOCK);
        if (probe_fd < 0) {
            return false;
        }
        close(probe_fd);
        return true;
    }

    bool is_ready() const override
    {
        return ready_;
    }

    void debug() const override
    {
        log(
            "joystick debug: available=" + std::string(available_ ? "true" : "false") +
            ", ready=" + std::string(ready_ ? "true" : "false"));
    }

    void start() override
    {
        stop();
        stop_flag_ = false;
        ready_ = false;
        available_ = false;
        expected_axes_ = 0;
        expected_buttons_ = 0;
        initialized_axes_.clear();
        initialized_buttons_.clear();
        thread_ = std::thread(&JoystickInputDriver::run, this);
    }

    void stop() override
    {
        stop_flag_ = true;
        if (thread_.joinable()) {
            thread_.join();
        }
        close_fd();
        available_ = false;
        ready_ = false;
    }

private:
    int fd_ = -1;
    std::thread thread_;
    std::atomic<bool> available_{false};
    std::atomic<bool> ready_{false};
    unsigned int expected_axes_ = 0;
    unsigned int expected_buttons_ = 0;
    std::set<unsigned int> initialized_axes_;
    std::set<unsigned int> initialized_buttons_;

    void run()
    {
        if (config_.device.empty()) {
            log("joystick " + config_.name + " has no device path");
            return;
        }

        fd_ = open(config_.device.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd_ < 0) {
            log("open joystick " + config_.device + " failed");
            return;
        }
        available_ = true;
        log("open joystick " + config_.device);

        unsigned char axis_count = 0;
        unsigned char button_count = 0;
        if (ioctl(fd_, JSIOCGAXES, &axis_count) == 0) {
            expected_axes_ = axis_count;
        }
        if (ioctl(fd_, JSIOCGBUTTONS, &button_count) == 0) {
            expected_buttons_ = button_count;
        }

        while (!stop_flag_) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd_, &fds);
            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = 100000;

            const int selected = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
            if (stop_flag_) {
                break;
            }
            if (selected == 0) {
                touch_runtime_sources();
                continue;
            }
            if (selected < 0) {
                if (errno == EINTR) {
                    continue;
                }
                log("joystick select failed: " + config_.name);
                break;
            }

            struct js_event event;
            const ssize_t length = read(fd_, &event, sizeof(event));
            if (length == sizeof(event)) {
                const bool initial = (event.type & JS_EVENT_INIT) != 0;
                const unsigned char event_type = event.type & ~JS_EVENT_INIT;
                if (event_type == JS_EVENT_AXIS) {
                    // The Linux joystick API can report -32767 for axes that
                    // have not been sampled yet after a wireless reconnect.
                    // Treat the initial snapshot as neutral so reconnecting a
                    // controller can never command motion.  A normal axis
                    // event replaces this value as soon as the stick moves.
                    set_signal(
                        "js.axis." + std::to_string(event.number),
                        initial ? 0.0 :
                            event.value / static_cast<double>(kAxisValueMax));
                    if (initial) {
                        initialized_axes_.insert(event.number);
                    }
                } else if (event_type == JS_EVENT_BUTTON) {
                    set_signal(
                        "js.button." + std::to_string(event.number),
                        event.value != 0 ? 1.0 : 0.0);
                    if (initial) {
                        initialized_buttons_.insert(event.number);
                    }
                }
                if (initial) {
                    mark_ready_if_initialized();
                } else if (!ready_) {
                    // Some joystick implementations omit JS_EVENT_INIT.  In
                    // that case the driver explicitly declares the first
                    // valid event sufficient for its initial snapshot.
                    ready_ = true;
                    log("joystick " + config_.name + " ready after first valid event");
                }
                continue;
            }

            if (length < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
                touch_runtime_sources();
                continue;
            }
            log("joystick lost: " + config_.name);
            break;
        }

        available_ = false;
        close_fd();
    }

    void mark_ready_if_initialized()
    {
        if (ready_) {
            return;
        }
        if (initialized_axes_.size() < expected_axes_ ||
            initialized_buttons_.size() < expected_buttons_) {
            return;
        }
        ready_ = true;
        log("joystick " + config_.name + " initial state ready");
    }

    void touch_runtime_sources()
    {
        const std::lock_guard<std::mutex> guard(mapper_lock_);
        mapper_.touch_runtime_sources(config_.raw_sources);
    }

    void close_fd()
    {
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }
};

}  // namespace

std::unique_ptr<InputDriver> create_joystick_input_driver(
    const InputDeviceConfig &config,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler)
{
    return std::unique_ptr<InputDriver>(new JoystickInputDriver(
        config,
        mapper,
        mapper_lock,
        std::move(output_handler),
        std::move(log_handler)));
}

}  // namespace remote_controller
