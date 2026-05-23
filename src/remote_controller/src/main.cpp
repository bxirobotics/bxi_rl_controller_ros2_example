#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cctype>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "remote_controller/config.hpp"
#include "remote_controller/input_driver.hpp"
#include "remote_controller/input_mapper.hpp"

using namespace std::chrono_literals;
using remote_controller::InputMapper;
using remote_controller::RemoteConfig;

namespace {

struct ConfigFileStamp {
    bool valid = false;
    std::time_t mtime_sec = 0;
    long mtime_nsec = 0;
    off_t size = 0;

    bool operator==(const ConfigFileStamp &other) const
    {
        return valid == other.valid &&
               mtime_sec == other.mtime_sec &&
               mtime_nsec == other.mtime_nsec &&
               size == other.size;
    }

    bool operator!=(const ConfigFileStamp &other) const
    {
        return !(*this == other);
    }
};

ConfigFileStamp read_config_file_stamp(const std::string &path)
{
    struct stat stat_buffer;
    if (stat(path.c_str(), &stat_buffer) != 0) {
        return {};
    }

    ConfigFileStamp stamp;
    stamp.valid = true;
    stamp.mtime_sec = stat_buffer.st_mtim.tv_sec;
    stamp.mtime_nsec = stat_buffer.st_mtim.tv_nsec;
    stamp.size = stat_buffer.st_size;
    return stamp;
}

bool parse_bool_text(const std::string &text, bool &value)
{
    std::string normalized;
    normalized.reserve(text.size());
    for (const char ch : text) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }

    if (normalized == "true" || normalized == "1" || normalized == "yes" || normalized == "on") {
        value = true;
        return true;
    }
    if (normalized == "false" || normalized == "0" || normalized == "no" || normalized == "off") {
        value = false;
        return true;
    }
    return false;
}

}  // namespace

class COMPublisher : public rclcpp::Node {
public:
    COMPublisher(
        const std::string &config_path,
        const std::string &driver_type,
        bool config_hot_reload_enabled)
        : Node("COM_publisher"),
          config_path_(config_path),
          config_hot_reload_enabled_(config_hot_reload_enabled),
          mapper_(remote_controller::load_remote_config(config_path))
    {
        config_stamp_ = read_config_file_stamp(config_path_);
        next_config_check_ = std::chrono::steady_clock::now() + config_check_interval_;
        print_config_diagnostics(mapper_.config());
        com_pub_ = this->create_publisher<communication::msg::MotionCommands>(
            "motion_commands", 20);
        timer_ = this->create_wall_timer(10ms, std::bind(&COMPublisher::timer_callback, this));

        input_driver_ = remote_controller::create_input_driver(
            driver_type,
            mapper_,
            lock_,
            [this](const std::vector<std::string> &outputs) { dispatch_outputs(outputs); },
            [this](const std::string &message) {
                RCLCPP_INFO(this->get_logger(), "%s", message.c_str());
            });
        input_driver_->start();
    }

    ~COMPublisher()
    {
        if (input_driver_) {
            input_driver_->stop();
        }
    }

private:
    mutable std::mutex lock_;
    std::string config_path_;
    bool config_hot_reload_enabled_ = true;
    ConfigFileStamp config_stamp_;
    const std::chrono::milliseconds config_check_interval_{1000};
    std::chrono::steady_clock::time_point next_config_check_;
    InputMapper mapper_;
    std::unique_ptr<remote_controller::InputDriver> input_driver_;
    std::map<std::string, bool> system_mutex_locked_;
    bool has_last_published_payload_ = false;
    communication::msg::MotionCommands last_published_payload_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<communication::msg::MotionCommands>::SharedPtr com_pub_;

    void print_config_diagnostics(const RemoteConfig &config)
    {
        for (const auto &diagnostic : config.diagnostics) {
            if (diagnostic.severity == "warning") {
                RCLCPP_WARN(this->get_logger(), "%s", diagnostic.message.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "%s", diagnostic.message.c_str());
            }
        }
    }

    void timer_callback()
    {
        if (config_hot_reload_enabled_) {
            check_config_reload();
        }

        auto message = communication::msg::MotionCommands();
        std::vector<std::string> outputs;
        bool publish_on_change = true;
        {
            const std::lock_guard<std::mutex> guard(lock_);
            outputs = mapper_.tick();
            mapper_.fill_message(message);
            publish_on_change = mapper_.config().publish_on_change;
        }
        dispatch_outputs(outputs);
        if (publish_on_change) {
            if (has_last_published_payload_ && message == last_published_payload_) {
                return;
            }

            last_published_payload_ = message;
            has_last_published_payload_ = true;
        }

        message.header.stamp = this->now();
        message.header.frame_id = "remote_controller";
        com_pub_->publish(message);
    }

    void check_config_reload()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_config_check_) {
            return;
        }
        next_config_check_ = now + config_check_interval_;

        const ConfigFileStamp stamp = read_config_file_stamp(config_path_);
        if (!stamp.valid) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                5000,
                "remote config is not readable: %s",
                config_path_.c_str());
            return;
        }
        if (stamp == config_stamp_) {
            return;
        }

        RemoteConfig config;
        try {
            config = remote_controller::load_remote_config(config_path_);
        } catch (const std::exception &exc) {
            RCLCPP_ERROR(
                this->get_logger(),
                "remote config hot reload failed: %s",
                exc.what());
            return;
        }

        std::string previous_js_device;
        {
            const std::lock_guard<std::mutex> guard(lock_);
            previous_js_device = mapper_.config().js_device;
            mapper_.reload_config(config);
            prune_system_mutexes_locked();
            has_last_published_payload_ = false;
            config_stamp_ = stamp;
        }

        print_config_diagnostics(config);
        RCLCPP_INFO(this->get_logger(), "remote config hot reloaded: %s", config_path_.c_str());
        if (previous_js_device != config.js_device) {
            RCLCPP_WARN(
                this->get_logger(),
                "js device changed from '%s' to '%s'; restart remote_controller to reopen the joystick device",
                previous_js_device.c_str(),
                config.js_device.c_str());
        }
    }

    void dispatch_outputs(const std::vector<std::string> &outputs)
    {
        for (const auto &output : outputs) {
            if (remote_controller::starts_with(output, "system.")) {
                run_system_action(output.substr(std::string("system.").size()));
            } else if (!output.empty()) {
                RCLCPP_WARN(this->get_logger(), "unknown binding output: %s", output.c_str());
            }
        }
    }

    void run_system_action(const std::string &action)
    {
        std::vector<std::string> commands;
        bool reset_motion_after_system = false;
        {
            const std::lock_guard<std::mutex> guard(lock_);
            const std::string blocked_by = blocking_system_mutex_locked(action);
            if (!blocked_by.empty()) {
                RCLCPP_WARN(
                    this->get_logger(),
                    "system.%s ignored because mutex '%s' is already acquired",
                    action.c_str(),
                    blocked_by.c_str());
                return;
            }

            const auto &system_commands = mapper_.config().system_commands;
            const auto command_it = system_commands.find(action);
            if (command_it == system_commands.end()) {
                RCLCPP_WARN(this->get_logger(), "unknown system output: system.%s", action.c_str());
                return;
            }
            commands = command_it->second;
            reset_motion_after_system = mapper_.config().reset_motion_after_system.count(action) > 0;
        }

        run_commands(commands);

        {
            const std::lock_guard<std::mutex> guard(lock_);
            update_system_mutexes_locked(action);
            if (reset_motion_after_system) {
                mapper_.reset_motion();
            }
        }
    }

    std::string blocking_system_mutex_locked(const std::string &action) const
    {
        for (const auto &mutex : mapper_.config().system_mutexes) {
            if (mutex.acquire != action) {
                continue;
            }
            const auto lock_it = system_mutex_locked_.find(mutex.name);
            if (lock_it != system_mutex_locked_.end() && lock_it->second) {
                return mutex.name;
            }
        }
        return "";
    }

    void update_system_mutexes_locked(const std::string &action)
    {
        for (const auto &mutex : mapper_.config().system_mutexes) {
            if (mutex.release == action) {
                system_mutex_locked_[mutex.name] = false;
            }
            if (mutex.acquire == action) {
                system_mutex_locked_[mutex.name] = true;
            }
        }
    }

    void prune_system_mutexes_locked()
    {
        std::map<std::string, bool> kept;
        for (const auto &mutex : mapper_.config().system_mutexes) {
            const auto lock_it = system_mutex_locked_.find(mutex.name);
            if (lock_it != system_mutex_locked_.end()) {
                kept[mutex.name] = lock_it->second;
            }
        }
        system_mutex_locked_ = kept;
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
    std::string driver_type = "joystick";
    std::string config_path;
    bool config_hot_reload_enabled = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--keyboard") {
            printf("in keyboard input mode\n");
            driver_type = "keyboard";
        } else if (arg == "--driver" && i + 1 < argc) {
            driver_type = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--hot-reload" && i + 1 < argc) {
            if (!parse_bool_text(argv[++i], config_hot_reload_enabled)) {
                fprintf(stderr, "--hot-reload expects true/false\n");
                return 1;
            }
        } else if (arg == "--no-hot-reload") {
            config_hot_reload_enabled = false;
        }
    }

    if (config_path.empty()) {
        fprintf(stderr, "remote_controller requires --config <yaml_path>\n");
        return 1;
    }

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<COMPublisher>(
        config_path,
        driver_type,
        config_hot_reload_enabled));
    rclcpp::shutdown();

    return 0;
}
