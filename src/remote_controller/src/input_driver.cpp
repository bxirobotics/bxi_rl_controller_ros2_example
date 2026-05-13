#include "remote_controller/input_driver.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <linux/joystick.h>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/select.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace remote_controller {
namespace {

class DriverBase : public InputDriver {
public:
    DriverBase(
        InputMapper &mapper,
        std::mutex &mapper_lock,
        DriverOutputHandler output_handler,
        DriverLogHandler log_handler)
        : mapper_(mapper),
          mapper_lock_(mapper_lock),
          output_handler_(std::move(output_handler)),
          log_handler_(std::move(log_handler))
    {
    }

protected:
    InputMapper &mapper_;
    std::mutex &mapper_lock_;
    DriverOutputHandler output_handler_;
    DriverLogHandler log_handler_;
    std::atomic<bool> stop_flag_{false};

    void log(const std::string &message) const
    {
        if (log_handler_) {
            log_handler_(message);
        }
    }

    void dispatch(const std::vector<std::string> &outputs) const
    {
        if (!outputs.empty() && output_handler_) {
            output_handler_(outputs);
        }
    }
};

class JoystickInputDriver : public DriverBase {
public:
    using DriverBase::DriverBase;

    ~JoystickInputDriver() override
    {
        stop();
    }

    std::string name() const override
    {
        return "joystick";
    }

    void start() override
    {
        stop_flag_ = false;
        open_loop();
        thread_ = std::thread(&JoystickInputDriver::run, this);
    }

    void stop() override
    {
        stop_flag_ = true;
        if (thread_.joinable()) {
            thread_.join();
        }
        close_fd();
    }

private:
    int fd_ = -1;
    std::thread thread_;

    void open_loop()
    {
        while (!stop_flag_) {
            fd_ = open(mapper_.config().js_device.c_str(), O_RDONLY | O_NONBLOCK);
            if (fd_ >= 0) {
                log("open js dev: " + mapper_.config().js_device);
                return;
            }
            log("open " + mapper_.config().js_device + " failed");
            sleep(1);
        }
    }

    void run()
    {
        while (!stop_flag_) {
            if (fd_ < 0) {
                open_loop();
                continue;
            }

            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd_, &fds);
            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = 100000;

            const int ready = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
            if (stop_flag_) {
                break;
            }
            if (ready == 0) {
                touch_runtime_sources();
                continue;
            }
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                log("js select failed, retry");
                close_fd();
                open_loop();
                continue;
            }

            struct js_event event;
            const ssize_t len = read(fd_, &event, sizeof(event));

            if (len == sizeof(event)) {
                std::vector<std::string> outputs;
                {
                    const std::lock_guard<std::mutex> guard(mapper_lock_);
                    if (event.type & JS_EVENT_AXIS) {
                        outputs = mapper_.set_axis(event.number, event.value);
                    } else if (event.type & JS_EVENT_BUTTON) {
                        outputs = mapper_.handle_button(event.number, event.value != 0);
                    } else {
                        log("unknown joystick event: " + std::to_string(event.type));
                    }
                }
                dispatch(outputs);
                continue;
            }

            if (len < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                    touch_runtime_sources();
                    continue;
                }
            }

            log("js dev lost, retry");
            close_fd();
            open_loop();
        }
    }

    void close_fd()
    {
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    void touch_runtime_sources()
    {
        const std::lock_guard<std::mutex> guard(mapper_lock_);
        mapper_.touch_runtime_sources_with_prefix("js.");
    }
};

class KeyboardInputDriver : public DriverBase {
public:
    using DriverBase::DriverBase;

    ~KeyboardInputDriver() override
    {
        stop();
    }

    std::string name() const override
    {
        return "keyboard";
    }

    void start() override
    {
        stop_flag_ = false;
        print_help();
        thread_ = std::thread(&KeyboardInputDriver::run, this);
    }

    void stop() override
    {
        stop_flag_ = true;
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::thread thread_;
    struct termios orig_termios_{};

    void print_help() const
    {
        const auto &keyboard = mapper_.config().keyboard;
        log("Keyboard mode enabled.");
        log(std::string("  ") + keyboard.forward + "/" + keyboard.backward + " : forward / backward");
        log(std::string("  ") + keyboard.yaw_left + "/" + keyboard.yaw_right + " : turn left / right");
        log(std::string("  ") + keyboard.strafe_left + "/" + keyboard.strafe_right + " : strafe left / right");
        log("  Space : full stop");
        for (const auto &item : keyboard.signals_by_key) {
            std::string line = std::string("  ") + item.first + " :";
            for (const auto &signal : item.second) {
                line += " " + signal;
            }
            log(line);
        }
        log("  ESC : exit");
    }

    void run()
    {
        const int tty_fd = isatty(STDIN_FILENO) ? STDIN_FILENO : open("/dev/tty", O_RDONLY);
        if (tty_fd < 0) {
            log("keyboard driver cannot open tty");
            return;
        }

        struct termios raw;
        tcgetattr(tty_fd, &orig_termios_);
        raw = orig_termios_;
        raw.c_lflag &= ~(tcflag_t)(ECHO | ICANON);
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        tcsetattr(tty_fd, TCSAFLUSH, &raw);

        using clock = std::chrono::steady_clock;
        std::map<char, clock::time_point> last_toggle_time;
        const auto debounce = std::chrono::milliseconds(300);

        while (!stop_flag_) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(tty_fd, &fds);
            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = mapper_.config().keyboard.poll_timeout_us;

            const int sel = select(tty_fd + 1, &fds, nullptr, nullptr, &tv);
            if (sel < 0) {
                break;
            }
            if (sel == 0) {
                continue;
            }

            char key;
            if (read(tty_fd, &key, 1) != 1) {
                break;
            }
            if (key == '\x1b') {
                break;
            }

            std::vector<std::string> outputs;
            {
                const std::lock_guard<std::mutex> guard(mapper_lock_);
                outputs = mapper_.handle_keyboard_key(key);
            }
            if (outputs.empty()) {
                continue;
            }

            const auto now = clock::now();
            const auto last_it = last_toggle_time.find(key);
            if (last_it != last_toggle_time.end() && (now - last_it->second) <= debounce) {
                continue;
            }
            last_toggle_time[key] = now;
            dispatch(outputs);
        }

        tcsetattr(tty_fd, TCSAFLUSH, &orig_termios_);
        if (tty_fd != STDIN_FILENO) {
            close(tty_fd);
        }
    }
};

}  // namespace

std::unique_ptr<InputDriver> create_input_driver(
    const std::string &driver_type,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler)
{
    if (driver_type == "keyboard") {
        return std::unique_ptr<InputDriver>(new KeyboardInputDriver(
            mapper,
            mapper_lock,
            std::move(output_handler),
            std::move(log_handler)));
    }
    if (driver_type == "joystick" || driver_type == "gamepad") {
        return std::unique_ptr<InputDriver>(new JoystickInputDriver(
            mapper,
            mapper_lock,
            std::move(output_handler),
            std::move(log_handler)));
    }
    throw std::runtime_error("unknown input driver type: " + driver_type);
}

}  // namespace remote_controller
