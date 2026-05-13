#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "remote_controller/config.hpp"
#include "remote_controller/input_driver.hpp"
#include "remote_controller/input_mapper.hpp"

using namespace std::chrono_literals;
using remote_controller::InputMapper;
using remote_controller::RemoteConfig;

class COMPublisher : public rclcpp::Node {
public:
    COMPublisher(const std::string &config_path, const std::string &driver_type)
        : Node("COM_publisher"),
          mapper_(remote_controller::load_remote_config(config_path))
    {
        print_config_diagnostics();
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
    InputMapper mapper_;
    std::unique_ptr<remote_controller::InputDriver> input_driver_;
    std::map<std::string, bool> system_mutex_locked_;
    bool has_last_published_payload_ = false;
    communication::msg::MotionCommands last_published_payload_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<communication::msg::MotionCommands>::SharedPtr com_pub_;

    void print_config_diagnostics()
    {
        for (const auto &diagnostic : mapper_.config().diagnostics) {
            if (diagnostic.severity == "warning") {
                RCLCPP_WARN(this->get_logger(), "%s", diagnostic.message.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "%s", diagnostic.message.c_str());
            }
        }
    }

    void timer_callback()
    {
        auto message = communication::msg::MotionCommands();
        std::vector<std::string> outputs;
        {
            const std::lock_guard<std::mutex> guard(lock_);
            outputs = mapper_.tick();
            mapper_.fill_message(message);
        }
        dispatch_outputs(outputs);
        if (mapper_.config().publish_on_change) {
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
        const std::string blocked_by = blocking_system_mutex(action);
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

        run_commands(command_it->second);
        update_system_mutexes(action);

        if (mapper_.config().reset_motion_after_system.count(action) > 0) {
            const std::lock_guard<std::mutex> guard(lock_);
            mapper_.reset_motion();
        }
    }

    std::string blocking_system_mutex(const std::string &action) const
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

    void update_system_mutexes(const std::string &action)
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

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--keyboard") {
            printf("in keyboard input mode\n");
            driver_type = "keyboard";
        } else if (arg == "--driver" && i + 1 < argc) {
            driver_type = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    if (config_path.empty()) {
        fprintf(stderr, "remote_controller requires --config <yaml_path>\n");
        return 1;
    }

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<COMPublisher>(config_path, driver_type));
    rclcpp::shutdown();

    return 0;
}
