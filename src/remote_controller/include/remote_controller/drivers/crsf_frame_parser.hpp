#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace remote_controller {

// Incremental decoder for CRSF frames.  It accepts arbitrarily fragmented
// serial reads and only reports a channel snapshot after the complete frame
// has passed CRC-8/DVB-S2 validation.
class CrsfFrameParser {
public:
    static constexpr std::uint8_t kSyncByte = 0xC8;
    static constexpr std::uint8_t kRcChannelsPackedFrameType = 0x16;
    static constexpr std::size_t kChannelCount = 16;
    static constexpr std::size_t kRcChannelsPayloadSize = 22;
    static constexpr std::size_t kMaxFrameLength = 64;

    using Channels = std::array<std::uint16_t, kChannelCount>;
    using ChannelsHandler = std::function<void(const Channels &)>;

    explicit CrsfFrameParser(ChannelsHandler channels_handler = ChannelsHandler())
        : channels_handler_(std::move(channels_handler))
    {
    }

    void reset()
    {
        buffer_.clear();
    }

    void push(const std::uint8_t *data, std::size_t size)
    {
        if (data == nullptr || size == 0) {
            return;
        }
        buffer_.insert(buffer_.end(), data, data + size);
        process_buffer();
    }

    void push(const std::vector<std::uint8_t> &data)
    {
        push(data.data(), data.size());
    }

    static std::uint8_t crc8_dvb_s2(const std::uint8_t *data, std::size_t size)
    {
        std::uint8_t crc = 0;
        while (size-- > 0) {
            crc ^= *data++;
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc & 0x80u) != 0 ?
                    static_cast<std::uint8_t>((crc << 1) ^ 0xD5u) :
                    static_cast<std::uint8_t>(crc << 1);
            }
        }
        return crc;
    }

    static bool decode_rc_channels(
        const std::uint8_t *payload,
        std::size_t payload_size,
        Channels &channels)
    {
        if (payload == nullptr || payload_size < kRcChannelsPayloadSize) {
            return false;
        }

        std::uint32_t bits = 0;
        std::uint8_t bit_count = 0;
        std::size_t payload_index = 0;
        for (std::size_t channel = 0; channel < kChannelCount; ++channel) {
            while (bit_count < 11) {
                if (payload_index >= payload_size) {
                    return false;
                }
                bits |= static_cast<std::uint32_t>(payload[payload_index++]) << bit_count;
                bit_count += 8;
            }
            channels[channel] = static_cast<std::uint16_t>(bits & 0x07FFu);
            bits >>= 11;
            bit_count -= 11;
        }
        return true;
    }

private:
    std::vector<std::uint8_t> buffer_;
    ChannelsHandler channels_handler_;

    void process_buffer()
    {
        while (!buffer_.empty()) {
            if (buffer_.front() != kSyncByte) {
                buffer_.erase(buffer_.begin());
                continue;
            }
            if (buffer_.size() < 2) {
                return;
            }

            const std::size_t frame_length = buffer_[1];
            if (frame_length < 3 || frame_length > kMaxFrameLength) {
                buffer_.erase(buffer_.begin());
                continue;
            }

            const std::size_t total_size = 2 + frame_length;
            if (buffer_.size() < total_size) {
                return;
            }

            const std::uint8_t crc_received = buffer_[total_size - 1];
            const std::uint8_t crc_calculated = crc8_dvb_s2(
                buffer_.data() + 2,
                frame_length - 1);
            if (crc_calculated != crc_received) {
                // Drop only the possible frame head so a later 0xC8 byte can
                // still become the beginning of the next frame.
                buffer_.erase(buffer_.begin());
                continue;
            }

            const std::uint8_t frame_type = buffer_[2];
            const std::uint8_t *payload = buffer_.data() + 3;
            const std::size_t payload_size = frame_length - 2;
            if (frame_type == kRcChannelsPackedFrameType && channels_handler_) {
                Channels channels{};
                if (decode_rc_channels(payload, payload_size, channels)) {
                    channels_handler_(channels);
                }
            }
            buffer_.erase(buffer_.begin(), buffer_.begin() + total_size);
        }
    }
};

}  // namespace remote_controller
