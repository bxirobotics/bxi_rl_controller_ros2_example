#include "remote_controller/drivers/driver_registry.hpp"

#include <map>
#include <utility>

#include "builtin_factories.hpp"

namespace remote_controller {
namespace {

std::map<std::string, InputDriverFactory> &driver_factories()
{
    static std::map<std::string, InputDriverFactory> factories;
    return factories;
}

std::mutex &driver_factories_lock()
{
    static std::mutex lock;
    return lock;
}

}  // namespace

bool register_input_driver_factory(const std::string &type, InputDriverFactory factory)
{
    if (type.empty() || !factory) {
        return false;
    }
    const std::lock_guard<std::mutex> guard(driver_factories_lock());
    if (driver_factories().count(type) > 0) {
        return false;
    }
    driver_factories()[type] = std::move(factory);
    return true;
}

bool has_input_driver_factory(const std::string &type)
{
    if (type == "keyboard" || type == "joystick" || type == "gamepad" || type == "crsf") {
        return true;
    }
    const std::lock_guard<std::mutex> guard(driver_factories_lock());
    return driver_factories().count(type) > 0;
}

std::unique_ptr<InputDriver> create_input_driver(
    const InputDeviceConfig &config,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler)
{
    if (config.type == "keyboard") {
        return create_keyboard_input_driver(
            config,
            mapper,
            mapper_lock,
            std::move(output_handler),
            std::move(log_handler));
    }
    if (config.type == "joystick" || config.type == "gamepad") {
        return create_joystick_input_driver(
            config,
            mapper,
            mapper_lock,
            std::move(output_handler),
            std::move(log_handler));
    }
    if (config.type == "crsf") {
        return create_crsf_input_driver(
            config,
            mapper,
            mapper_lock,
            std::move(output_handler),
            std::move(log_handler));
    }

    InputDriverFactory factory;
    {
        const std::lock_guard<std::mutex> guard(driver_factories_lock());
        const auto factory_it = driver_factories().find(config.type);
        if (factory_it == driver_factories().end()) {
            return nullptr;
        }
        factory = factory_it->second;
    }
    return factory(config, mapper, mapper_lock, std::move(output_handler), std::move(log_handler));
}

}  // namespace remote_controller
