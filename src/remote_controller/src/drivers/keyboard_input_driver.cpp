#include "remote_controller/drivers/input_driver.hpp"

#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/select.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace remote_controller {
namespace {

class KeyboardInputDriver : public InputDriverBase {
public:
    using InputDriverBase::InputDriverBase;

    ~KeyboardInputDriver() override
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
        if (isatty(STDIN_FILENO)) {
            return true;
        }
        const int tty_fd = open("/dev/tty", O_RDONLY | O_NONBLOCK);
        if (tty_fd < 0) {
            return false;
        }
        close(tty_fd);
        return true;
    }

    bool is_ready() const override
    {
        return ready_;
    }

    void debug() const override
    {
        log(
            "keyboard debug: available=" + std::string(available_ ? "true" : "false") +
            ", ready=" + std::string(ready_ ? "true" : "false"));
    }

    void start() override
    {
        stop();
        stop_flag_ = false;
        ready_ = false;
        available_ = false;
        thread_ = std::thread(&KeyboardInputDriver::run, this);
    }

    void stop() override
    {
        stop_flag_ = true;
        if (thread_.joinable()) {
            thread_.join();
        }
        available_ = false;
        ready_ = false;
    }

private:
    std::thread thread_;
    std::atomic<bool> available_{false};
    std::atomic<bool> ready_{false};
    struct termios orig_termios_ {};

    void run()
    {
        const int tty_fd = isatty(STDIN_FILENO) ? STDIN_FILENO : open("/dev/tty", O_RDONLY);
        if (tty_fd < 0) {
            log("keyboard driver cannot open tty");
            return;
        }

        struct termios raw;
        if (tcgetattr(tty_fd, &orig_termios_) != 0) {
            log("keyboard driver cannot read tty settings");
            if (tty_fd != STDIN_FILENO) {
                close(tty_fd);
            }
            return;
        }
        raw = orig_termios_;
        raw.c_lflag &= ~(tcflag_t)(ECHO | ICANON);
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(tty_fd, TCSAFLUSH, &raw) != 0) {
            log("keyboard driver cannot set raw tty mode");
            if (tty_fd != STDIN_FILENO) {
                close(tty_fd);
            }
            return;
        }

        available_ = true;
        ready_ = true;  // Keyboard declares an empty initial snapshot safe.
        log("keyboard " + config_.name + " ready");

        using clock = std::chrono::steady_clock;
        std::map<std::string, clock::time_point> expiry;
        const KeyboardConfig &keyboard = config_.keyboard;
        while (!stop_flag_) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(tty_fd, &fds);
            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = keyboard.poll_timeout_us;
            const int selected = select(tty_fd + 1, &fds, nullptr, nullptr, &tv);
            if (selected < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }

            const auto now = clock::now();
            for (auto it = expiry.begin(); it != expiry.end();) {
                if (it->second <= now) {
                    set_signal(it->first, 0.0);
                    it = expiry.erase(it);
                } else {
                    ++it;
                }
            }

            if (selected == 0) {
                continue;
            }

            char key = '\0';
            if (read(tty_fd, &key, 1) != 1) {
                break;
            }
            if (key == '\x1b') {
                break;
            }
            handle_key(key, expiry, now);
        }

        tcsetattr(tty_fd, TCSAFLUSH, &orig_termios_);
        if (tty_fd != STDIN_FILENO) {
            close(tty_fd);
        }
        available_ = false;
        ready_ = false;
    }

    void handle_key(
        char key,
        std::map<std::string, std::chrono::steady_clock::time_point> &expiry,
        std::chrono::steady_clock::time_point now)
    {
        const KeyboardConfig &keyboard = config_.keyboard;
        const auto schedule = [&keyboard, &expiry, now](const std::string &signal) {
            const auto hold_it = keyboard.hold_ms_by_signal.find(signal);
            const int hold_ms = hold_it == keyboard.hold_ms_by_signal.end() ?
                keyboard.hold_ms : hold_it->second;
            expiry[signal] = now + std::chrono::milliseconds(hold_ms);
        };

        if (key == keyboard.forward) {
            set_signal(keyboard.vx_source, -1.0);
            schedule(keyboard.vx_source);
            return;
        }
        if (key == keyboard.backward) {
            set_signal(keyboard.vx_source, 1.0);
            schedule(keyboard.vx_source);
            return;
        }
        if (key == keyboard.yaw_left) {
            set_signal(keyboard.yaw_source, -1.0);
            schedule(keyboard.yaw_source);
            return;
        }
        if (key == keyboard.yaw_right) {
            set_signal(keyboard.yaw_source, 1.0);
            schedule(keyboard.yaw_source);
            return;
        }
        if (key == keyboard.strafe_left) {
            set_signal(keyboard.vy_source, -1.0);
            schedule(keyboard.vy_source);
            return;
        }
        if (key == keyboard.strafe_right) {
            set_signal(keyboard.vy_source, 1.0);
            schedule(keyboard.vy_source);
            return;
        }
        if (key == keyboard.stop) {
            set_signal(keyboard.vx_source, 0.0);
            set_signal(keyboard.vy_source, 0.0);
            set_signal(keyboard.yaw_source, 0.0);
            return;
        }

        const auto signal_it = keyboard.signals_by_key.find(key);
        if (signal_it == keyboard.signals_by_key.end()) {
            return;
        }
        for (const auto &signal : signal_it->second) {
            set_signal(signal, 1.0);
            schedule(signal);
        }
    }
};

}  // namespace

std::unique_ptr<InputDriver> create_keyboard_input_driver(
    const InputDeviceConfig &config,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler)
{
    return std::unique_ptr<InputDriver>(new KeyboardInputDriver(
        config,
        mapper,
        mapper_lock,
        std::move(output_handler),
        std::move(log_handler)));
}

}  // namespace remote_controller
