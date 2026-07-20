#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

#include "remote_controller/drivers/crsf_frame_parser.hpp"

namespace {

using remote_controller::CrsfFrameParser;

std::vector<std::uint8_t> make_channels_frame(const CrsfFrameParser::Channels &channels)
{
    std::array<std::uint8_t, CrsfFrameParser::kRcChannelsPayloadSize> payload{};
    for (std::size_t channel = 0; channel < channels.size(); ++channel) {
        const std::size_t bit_offset = channel * 11;
        const std::size_t byte_offset = bit_offset / 8;
        const std::size_t shift = bit_offset % 8;
        const std::uint32_t packed =
            static_cast<std::uint32_t>(channels[channel] & 0x07FFu) << shift;
        payload[byte_offset] |= static_cast<std::uint8_t>(packed & 0xFFu);
        if (byte_offset + 1 < payload.size()) {
            payload[byte_offset + 1] |= static_cast<std::uint8_t>((packed >> 8) & 0xFFu);
        }
        if (byte_offset + 2 < payload.size()) {
            payload[byte_offset + 2] |= static_cast<std::uint8_t>((packed >> 16) & 0xFFu);
        }
    }

    constexpr std::uint8_t frame_length =
        1 + CrsfFrameParser::kRcChannelsPayloadSize + 1;
    std::vector<std::uint8_t> frame(2 + frame_length);
    frame[0] = CrsfFrameParser::kSyncByte;
    frame[1] = frame_length;
    frame[2] = CrsfFrameParser::kRcChannelsPackedFrameType;
    for (std::size_t index = 0; index < payload.size(); ++index) {
        frame[3 + index] = payload[index];
    }
    frame.back() = CrsfFrameParser::crc8_dvb_s2(frame.data() + 2, frame_length - 1);
    return frame;
}

void test_fragmented_frame_and_channel_unpacking()
{
    CrsfFrameParser::Channels expected{};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<std::uint16_t>((174 + index * 97) & 0x07FFu);
    }
    const std::vector<std::uint8_t> frame = make_channels_frame(expected);

    int callback_count = 0;
    CrsfFrameParser::Channels received{};
    CrsfFrameParser parser([&callback_count, &received](const CrsfFrameParser::Channels &channels) {
        ++callback_count;
        received = channels;
    });

    const std::vector<std::uint8_t> noise{0x00, 0x7E, 0xC7};
    parser.push(noise);
    parser.push(frame.data(), 5);
    assert(callback_count == 0);
    parser.push(frame.data() + 5, frame.size() - 5);
    assert(callback_count == 1);
    assert(received == expected);
}

void test_bad_crc_is_ignored_and_parser_resynchronizes()
{
    CrsfFrameParser::Channels expected{};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<std::uint16_t>(992 + index);
    }
    std::vector<std::uint8_t> corrupt = make_channels_frame(expected);
    corrupt.back() ^= 0x55u;
    const std::vector<std::uint8_t> valid = make_channels_frame(expected);
    corrupt.insert(corrupt.end(), valid.begin(), valid.end());

    int callback_count = 0;
    CrsfFrameParser parser([&callback_count](const CrsfFrameParser::Channels &) {
        ++callback_count;
    });
    parser.push(corrupt);
    assert(callback_count == 1);
}

}  // namespace

int main()
{
    test_fragmented_frame_and_channel_unpacking();
    test_bad_crc_is_ignored_and_parser_resynchronizes();
    return 0;
}
