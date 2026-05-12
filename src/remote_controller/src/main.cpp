#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <linux/joystick.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <sys/select.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "remote_controller/config.hpp"
#include "remote_controller/input_mapper.hpp"

using namespace std::chrono_literals;
using remote_controller::InputMapper;
using remote_controller::RemoteConfig;

class COMPublisher : public rclcpp::Node {
public:
    COMPublisher(const std::string &config_path, bool use_keyboard)
        : Node("COM_publisher"),
          mapper_(remote_controller::load_remote_config(config_path)),
          use_keyboard_(use_keyboard)
    {
        com_pub_ = this->create_publisher<communication::msg::MotionCommands>(
            "motion_commands", 20);
        timer_ = this->create_wall_timer(10ms, std::bind(&COMPublisher::timer_callback, this));

        if (use_keyboard_) {
            print_keyboard_help();
            kb_loop_thread_ = std::thread(&COMPublisher::kb_loop, this);
        } else {
            open_joystick_loop();
            js_loop_thread_ = std::thread(&COMPublisher::js_loop, this);
        }
    }

    ~COMPublisher()
    {
        stop_flag_ = true;
        if (js_fd_ >= 0) {
            close(js_fd_);
            js_fd_ = -1;
        }
        if (kb_loop_thread_.joinable()) {
            kb_loop_thread_.join();
        }
        if (js_loop_thread_.joinable()) {
            js_loop_thread_.join();
        }
    }

private:
    mutable std::mutex lock_;
    InputMapper mapper_;
    bool use_keyboard_ = false;
    std::atomic<bool> stop_flag_{false};

    int js_fd_ = -1;
    std::thread js_loop_thread_;
    std::thread kb_loop_thread_;
    struct termios orig_termios_{};
    bool launch_lock_ = false;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<communication::msg::MotionCommands>::SharedPtr com_pub_;

    void timer_callback()
    {
        auto message = communication::msg::MotionCommands();
        {
            const std::lock_guard<std::mutex> guard(lock_);
            mapper_.fill_message(message);
        }
        com_pub_->publish(message);
    }

    void open_joystick_loop()
    {
        while (!stop_flag_) {
            js_fd_ = open(mapper_.config().js_device.c_str(), O_RDONLY);
            if (js_fd_ >= 0) {
                printf("open js dev: %s\n", mapper_.config().js_device.c_str());
                return;
            }
            printf("open:%s failed\n", mapper_.config().js_device.c_str());
            sleep(1);
        }
    }

    void js_loop()
    {
        while (!stop_flag_) {
            struct js_event event;
            const ssize_t len = read(js_fd_, &event, sizeof(event));

            if (len == sizeof(event)) {
                std::vector<std::string> outputs;
                {
                    const std::lock_guard<std::mutex> guard(lock_);
                    if (event.type & JS_EVENT_AXIS) {
                        outputs = mapper_.set_axis(event.number, event.value);
                    } else if (event.type & JS_EVENT_BUTTON) {
                        outputs = mapper_.handle_button(event.number, event.value != 0);
                    } else {
                        printf("unknown event:%u\n", event.type);
                    }
                }
                dispatch_outputs(outputs);
                continue;
            }

            if (stop_flag_) {
                break;
            }

            if (len <= 0) {
                printf("js dev lost, retry\n");
                if (js_fd_ >= 0) {
                    close(js_fd_);
                    js_fd_ = -1;
                }
                open_joystick_loop();
            }
        }
    }

    void kb_loop()
    {
        const int tty_fd = isatty(STDIN_FILENO) ? STDIN_FILENO : open("/dev/tty", O_RDONLY);
        if (tty_fd < 0) {
            printf("kb_loop: cannot open tty\n");
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
            tv.tv_usec = mapper_.config().keyboard.timeout_us;

            const int sel = select(tty_fd + 1, &fds, nullptr, nullptr, &tv);
            if (sel < 0) {
                break;
            }
            if (sel == 0) {
                const std::lock_guard<std::mutex> guard(lock_);
                mapper_.zero_motion_axes();
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
                const std::lock_guard<std::mutex> guard(lock_);
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
            dispatch_outputs(outputs);
        }

        tcsetattr(tty_fd, TCSAFLUSH, &orig_termios_);
        if (tty_fd != STDIN_FILENO) {
            close(tty_fd);
        }
    }

    void print_keyboard_help()
    {
        const auto &keyboard = mapper_.config().keyboard;
        printf("Keyboard mode enabled.\n");
        printf("  %c/%c      : forward / backward\n", keyboard.forward, keyboard.backward);
        printf("  %c/%c      : turn left / right\n", keyboard.yaw_left, keyboard.yaw_right);
        printf("  %c/%c      : strafe left / right\n", keyboard.strafe_left, keyboard.strafe_right);
        printf("  Space    : full stop\n");
        for (const auto &item : keyboard.bindings) {
            printf("  %c        : %s\n", item.first, item.second.c_str());
        }
        printf("  ESC      : exit\n");
    }

    void dispatch_outputs(const std::vector<std::string> &outputs)
    {
        for (const auto &output : outputs) {
            bool consumed = false;
            {
                const std::lock_guard<std::mutex> guard(lock_);
                consumed = mapper_.apply_button_output(output);
            }
            if (consumed) {
                continue;
            }

            if (output == "system.stop") {
                stop_robot_processes();
            } else if (output == "system.start") {
                start_robot_processes();
            } else if (!output.empty()) {
                RCLCPP_WARN(this->get_logger(), "unknown binding output: %s", output.c_str());
            }
        }
    }

    void stop_robot_processes()
    {
        run_commands(mapper_.config().stop_commands);
        launch_lock_ = false;
        const std::lock_guard<std::mutex> guard(lock_);
        mapper_.reset_motion();
    }

    void start_robot_processes()
    {
        if (launch_lock_) {
            printf("\nprograme already exist! stop launch!!\n\n");
            return;
        }

        run_commands(mapper_.config().start_commands);
        printf("run robot\n");
        launch_lock_ = true;
        const std::lock_guard<std::mutex> guard(lock_);
        mapper_.reset_motion();
    }

    void run_commands(const std::vector<std::string> &commands)
    {
        for (const auto &command : commands) {
            const int ret = std::system(command.c_str());
            (void)ret;
        }
    }
};

int main(int argc, const char *argv[])
{
    bool use_keyboard = false;
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--keyboard") {
            printf("in keyboard input mode\n");
            use_keyboard = true;
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    if (config_path.empty()) {
        fprintf(stderr, "remote_controller requires --config <yaml_path>\n");
        return 1;
    }

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<COMPublisher>(config_path, use_keyboard));
    rclcpp::shutdown();

    return 0;
}
