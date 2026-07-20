#include "remote_controller/drivers/input_driver.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/select.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "remote_controller/drivers/crsf_frame_parser.hpp"

namespace remote_controller {
namespace {

constexpr int kDefaultBaudRate = 460800;
constexpr int kDefaultChannelMin = 174;
constexpr int kDefaultChannelMax = 1811;
constexpr std::chrono::milliseconds kRetryInterval(100);

speed_t baud_to_termios_speed(int baud_rate)
{
    switch (baud_rate) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
#ifdef B230400
    case 230400: return B230400;
#endif
#ifdef B460800
    case 460800: return B460800;
#endif
#ifdef B500000
    case 500000: return B500000;
#endif
#ifdef B921600
    case 921600: return B921600;
#endif
    default:
        throw std::runtime_error("unsupported CRSF baud_rate: " + std::to_string(baud_rate));
    }
}

int option_int(
    const InputDeviceConfig &config,
    const std::string &name,
    int fallback,
    const std::string &legacy_name = "")
{
    const auto find_option = [&config](const std::string &option_name) {
        return config.options.find(option_name);
    };
    auto option = find_option(name);
    if (option == config.options.end() && !legacy_name.empty()) {
        option = find_option(legacy_name);
    }
    if (option == config.options.end()) {
        return fallback;
    }

    std::size_t parsed = 0;
    const int value = std::stoi(option->second, &parsed);
    if (parsed != option->second.size()) {
        throw std::runtime_error(
            "CRSF option '" + name + "' must be an integer: " + option->second);
    }
    return value;
}

std::int64_t steady_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

class CrsfInputDriver : public InputDriverBase {
public:
    CrsfInputDriver(
        InputDeviceConfig config,
        InputMapper &mapper,
        std::mutex &mapper_lock,
        DriverOutputHandler output_handler,
        DriverLogHandler log_handler)
        : InputDriverBase(
              std::move(config),
              mapper,
              mapper_lock,
              std::move(output_handler),
              std::move(log_handler)),
          baud_rate_(option_int(config_, "baud_rate", kDefaultBaudRate, "baudrate")),
          channel_min_(option_int(config_, "channel_min", kDefaultChannelMin)),
          channel_max_(option_int(config_, "channel_max", kDefaultChannelMax)),
          probe_parser_([this](const CrsfFrameParser::Channels &channels) {
              handle_probe_frame(channels);
          }),
          active_parser_([this](const CrsfFrameParser::Channels &channels) {
              handle_active_frame(channels);
          })
    {
        if (config_.device.empty()) {
            throw std::runtime_error("CRSF input '" + config_.name + "' has no device path");
        }
        if (channel_min_ >= channel_max_) {
            throw std::runtime_error(
                "CRSF channel_min must be smaller than channel_max for input '" +
                config_.name + "'");
        }
        (void)baud_to_termios_speed(baud_rate_);
    }

    ~CrsfInputDriver() override
    {
        stop();
    }

    std::string name() const override
    {
        return config_.name;
    }

    bool is_available() override
    {
        if (started_) {
            return reader_open_ && has_recent_valid_frame();
        }

        refresh_probe();
        return probe_open_ && has_recent_valid_frame();
    }

    bool availability_handles_loss_timeout() const override
    {
        return true;
    }

    bool is_ready() const override
    {
        return ready_;
    }

    void debug() const override
    {
        std::array<double, CrsfFrameParser::kChannelCount> channels{};
        bool has_channels = false;
        {
            const std::lock_guard<std::mutex> guard(debug_lock_);
            has_channels = has_debug_channels_;
            channels = debug_channels_;
        }

        if (!has_channels) {
            log("CRSF debug: no CRC-correct RC channel frame received yet");
            return;
        }

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << "CRSF channels:";
        for (std::size_t index = 0; index < channels.size(); ++index) {
            stream << " ch" << (index + 1) << '=' << channels[index];
        }
        log(stream.str());
    }

    void start() override
    {
        stop();
        {
            const std::lock_guard<std::mutex> guard(io_lock_);
            close_fd(probe_fd_);
            probe_open_ = false;
            probe_parser_.reset();
        }
        last_valid_frame_ns_ = 0;
        ready_ = false;
        reader_open_ = false;
        clear_debug_channels();
        stop_flag_ = false;
        started_ = true;
        thread_ = std::thread(&CrsfInputDriver::run, this);
    }

    void stop() override
    {
        stop_flag_ = true;
        if (thread_.joinable()) {
            thread_.join();
        }
        started_ = false;
        reader_open_ = false;
        ready_ = false;
        last_valid_frame_ns_ = 0;
        clear_debug_channels();
        const std::lock_guard<std::mutex> guard(io_lock_);
        close_fd(active_fd_);
        close_fd(probe_fd_);
        probe_open_ = false;
        probe_parser_.reset();
        active_parser_.reset();
    }

private:
    const int baud_rate_;
    const int channel_min_;
    const int channel_max_;
    std::thread thread_;
    std::mutex io_lock_;
    mutable std::mutex debug_lock_;
    int probe_fd_ = -1;
    int active_fd_ = -1;
    CrsfFrameParser probe_parser_;
    CrsfFrameParser active_parser_;
    std::atomic<bool> started_{false};
    std::atomic<bool> reader_open_{false};
    std::atomic<bool> probe_open_{false};
    std::atomic<bool> ready_{false};
    std::atomic<std::int64_t> last_valid_frame_ns_{0};
    std::array<double, CrsfFrameParser::kChannelCount> debug_channels_{};
    bool has_debug_channels_ = false;

    void record_valid_frame()
    {
        last_valid_frame_ns_ = steady_now_ns();
    }

    bool has_recent_valid_frame() const
    {
        const std::int64_t last_valid_frame = last_valid_frame_ns_.load();
        if (last_valid_frame == 0) {
            return false;
        }
        const std::int64_t age_ns = steady_now_ns() - last_valid_frame;
        const std::int64_t timeout_ns =
            static_cast<std::int64_t>(config_.loss_timeout_ms) * 1000000LL;
        return age_ns >= 0 && age_ns <= timeout_ns;
    }

    void clear_debug_channels()
    {
        const std::lock_guard<std::mutex> guard(debug_lock_);
        debug_channels_.fill(0.0);
        has_debug_channels_ = false;
    }

    void cache_debug_channels(const CrsfFrameParser::Channels &channels)
    {
        const std::lock_guard<std::mutex> guard(debug_lock_);
        for (std::size_t index = 0; index < channels.size(); ++index) {
            debug_channels_[index] = normalize_channel(channels[index]);
        }
        has_debug_channels_ = true;
    }

    int open_serial() const
    {
        const int fd = open(config_.device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd < 0) {
            return -1;
        }
        if (!configure_serial(fd)) {
            close(fd);
            return -1;
        }
        return fd;
    }

    bool configure_serial(int fd) const
    {
        struct termios settings {};
        if (tcgetattr(fd, &settings) != 0) {
            return false;
        }
        cfmakeraw(&settings);
        settings.c_cflag |= static_cast<tcflag_t>(CLOCAL | CREAD);
        settings.c_cflag &= static_cast<tcflag_t>(~PARENB);
        settings.c_cflag &= static_cast<tcflag_t>(~CSTOPB);
        settings.c_cflag &= static_cast<tcflag_t>(~CSIZE);
        settings.c_cflag |= CS8;
#ifdef CRTSCTS
        settings.c_cflag &= static_cast<tcflag_t>(~CRTSCTS);
#endif
        settings.c_iflag &= static_cast<tcflag_t>(~(IXON | IXOFF | IXANY));
        settings.c_cc[VMIN] = 0;
        settings.c_cc[VTIME] = 0;

        const speed_t speed = baud_to_termios_speed(baud_rate_);
        if (cfsetispeed(&settings, speed) != 0 || cfsetospeed(&settings, speed) != 0) {
            return false;
        }
        if (tcsetattr(fd, TCSANOW, &settings) != 0) {
            return false;
        }
        tcflush(fd, TCIFLUSH);
        return true;
    }

    static void close_fd(int &fd)
    {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }

    bool read_available(int fd, CrsfFrameParser &parser)
    {
        std::array<std::uint8_t, 512> bytes{};
        while (true) {
            const ssize_t count = read(fd, bytes.data(), bytes.size());
            if (count > 0) {
                parser.push(bytes.data(), static_cast<std::size_t>(count));
                continue;
            }
            // With VMIN=0/VTIME=0 a non-blocking tty can report a normal
            // no-data condition as a zero-length read.  Keep the descriptor
            // open so the next availability scan can receive the next CRSF
            // frame; valid-frame freshness remains the disconnect guard.
            if (count == 0) {
                return true;
            }
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
                return true;
            }
            return false;
        }
    }

    void refresh_probe()
    {
        const std::lock_guard<std::mutex> guard(io_lock_);
        if (probe_fd_ < 0) {
            probe_parser_.reset();
            last_valid_frame_ns_ = 0;
            probe_fd_ = open_serial();
            probe_open_ = probe_fd_ >= 0;
            if (probe_fd_ < 0) {
                return;
            }
        }

        if (!read_available(probe_fd_, probe_parser_)) {
            close_fd(probe_fd_);
            probe_open_ = false;
            last_valid_frame_ns_ = 0;
            probe_parser_.reset();
        }
    }

    void run()
    {
        while (!stop_flag_) {
            if (active_fd_ < 0) {
                active_parser_.reset();
                last_valid_frame_ns_ = 0;
                active_fd_ = open_serial();
                reader_open_ = active_fd_ >= 0;
                if (active_fd_ < 0) {
                    std::this_thread::sleep_for(kRetryInterval);
                    continue;
                }
                log("CRSF serial opened: " + config_.device);
            }

            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(active_fd_, &fds);
            struct timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 100000;
            const int selected = select(active_fd_ + 1, &fds, nullptr, nullptr, &timeout);
            if (stop_flag_) {
                break;
            }
            if (selected == 0) {
                continue;
            }
            if (selected < 0) {
                if (errno == EINTR) {
                    continue;
                }
                log("CRSF serial select failed: " + config_.name);
                close_fd(active_fd_);
                reader_open_ = false;
                ready_ = false;
                continue;
            }
            if (!read_available(active_fd_, active_parser_)) {
                log("CRSF serial lost: " + config_.name);
                close_fd(active_fd_);
                reader_open_ = false;
                ready_ = false;
            }
        }

        close_fd(active_fd_);
        reader_open_ = false;
        ready_ = false;
    }

    double normalize_channel(std::uint16_t value) const
    {
        const double midpoint = (static_cast<double>(channel_min_) + channel_max_) * 0.5;
        const double half_range = (static_cast<double>(channel_max_) - channel_min_) * 0.5;
        const double normalized = (static_cast<double>(value) - midpoint) / half_range;
        return std::max(-1.0, std::min(1.0, normalized));
    }

    void handle_active_frame(const CrsfFrameParser::Channels &channels)
    {
        record_valid_frame();
        cache_debug_channels(channels);

        std::vector<std::pair<std::string, double>> signals;
        signals.reserve(CrsfFrameParser::kChannelCount);
        for (std::size_t index = 0; index < CrsfFrameParser::kChannelCount; ++index) {
            const std::string source = "crsf.channel." + std::to_string(index + 1);
            if (config_.raw_sources.count(source) == 0) {
                continue;
            }
            signals.emplace_back(source, normalize_channel(channels[index]));
        }
        if (config_.raw_sources.count("crsf.connected") > 0) {
            signals.emplace_back("crsf.connected", 1.0);
        }
        set_signals(signals);
        ready_ = true;
    }

    void handle_probe_frame(const CrsfFrameParser::Channels &channels)
    {
        record_valid_frame();
        cache_debug_channels(channels);
    }
};

}  // namespace

std::unique_ptr<InputDriver> create_crsf_input_driver(
    const InputDeviceConfig &config,
    InputMapper &mapper,
    std::mutex &mapper_lock,
    DriverOutputHandler output_handler,
    DriverLogHandler log_handler)
{
    return std::unique_ptr<InputDriver>(new CrsfInputDriver(
        config,
        mapper,
        mapper_lock,
        std::move(output_handler),
        std::move(log_handler)));
}

}  // namespace remote_controller
