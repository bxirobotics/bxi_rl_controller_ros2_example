#include "bxi_depth_camera/types.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace bxi_depth_camera
{
namespace
{

TEST(ProfileParsing, AcceptsCommaAndCrossForms)
{
    const auto comma = parse_profile("640,480,30", "profile");
    EXPECT_EQ(comma.width, 640);
    EXPECT_EQ(comma.height, 480);
    EXPECT_EQ(comma.fps, 30);

    const auto cross = parse_profile("848x480x60", "profile");
    EXPECT_EQ(cross.width, 848);
    EXPECT_EQ(cross.height, 480);
    EXPECT_EQ(cross.fps, 60);
}

TEST(ProfileParsing, SupportsOnlyCompleteAutomaticProfile)
{
    EXPECT_TRUE(parse_profile("0,0,0", "profile").automatic());
    EXPECT_THROW(parse_profile("0,480,30", "profile"), std::invalid_argument);
    EXPECT_THROW(parse_profile("640,480", "profile"), std::invalid_argument);
}

TEST(NameValidation, ProducesStableTopicAndNodeTokens)
{
    EXPECT_EQ(camera_name_token("head_depth_camera"), "head_depth_camera");
    EXPECT_EQ(topic_token("CP0F4630000L"), "SN_CP0F4630000L");
    EXPECT_EQ(node_token("349422070502"), "camera_SN_349422070502");
    EXPECT_THROW(camera_name_token("head-depth-camera"), std::invalid_argument);
    EXPECT_THROW(topic_token("serial/with/slash"), std::invalid_argument);
}

TEST(ConfigValidation, EnforcesDependentStreams)
{
    CameraConfig config;
    config.enable_color = false;
    config.align_depth = true;
    EXPECT_THROW(config.validate(), std::invalid_argument);

    config.align_depth = false;
    config.enable_depth = false;
    config.pointcloud_enabled = true;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

TEST(ConfigValidation, RejectsInvalidRatesAndTimeouts)
{
    CameraConfig config;
    config.pointcloud_max_fps = 0.0;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.pointcloud_max_fps = 10.0;
    config.device_timeout_sec = -1.0;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

} // namespace
} // namespace bxi_depth_camera
