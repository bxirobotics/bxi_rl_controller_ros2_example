#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace
{

template<typename T>
void declare_and_get(const rclcpp::Node::SharedPtr& node, const std::string& name, T& value)
{
  node->declare_parameter<T>(name, value);
  node->get_parameter(name, value);
}

void fillCameraInfo(sensor_msgs::msg::CameraInfo& ci,
                    int width,
                    int height,
                    double fx,
                    double fy,
                    double cx,
                    double cy)
{
  ci.width = width;
  ci.height = height;
  ci.distortion_model = "plumb_bob";
  ci.d.assign(5, 0.0);

  std::fill(ci.k.begin(), ci.k.end(), 0.0);
  ci.k[0] = fx;
  ci.k[2] = cx;
  ci.k[4] = fy;
  ci.k[5] = cy;
  ci.k[8] = 1.0;

  std::fill(ci.r.begin(), ci.r.end(), 0.0);
  ci.r[0] = 1.0;
  ci.r[4] = 1.0;
  ci.r[8] = 1.0;

  std::fill(ci.p.begin(), ci.p.end(), 0.0);
  ci.p[0] = fx;
  ci.p[2] = cx;
  ci.p[5] = fy;
  ci.p[6] = cy;
  ci.p[10] = 1.0;
}

void resizeNearest16U(const uint16_t* src, int src_w, int src_h, int src_stride_px,
                      uint16_t* dst, int dst_w, int dst_h)
{
  const double sx = static_cast<double>(src_w) / static_cast<double>(dst_w);
  const double sy = static_cast<double>(src_h) / static_cast<double>(dst_h);

  for (int y = 0; y < dst_h; ++y) {
    const int yy = std::min(static_cast<int>(std::floor(y * sy)), src_h - 1);
    const uint16_t* src_row = src + yy * src_stride_px;
    uint16_t* dst_row = dst + y * dst_w;

    for (int x = 0; x < dst_w; ++x) {
      const int xx = std::min(static_cast<int>(std::floor(x * sx)), src_w - 1);
      dst_row[x] = src_row[xx];
    }
  }
}

uint16_t clampDepthMm(uint16_t depth_mm, uint16_t min_mm, uint16_t max_mm)
{
  if (depth_mm == 0) {
    return 0;
  }
  return std::max(min_mm, std::min(depth_mm, max_mm));
}

double clamp01(double value)
{
  return std::max(0.0, std::min(value, 1.0));
}

void applyHoleFillingFilter(std::vector<uint16_t>& depth_mm, int width, int height, int passes)
{
  for (int pass = 0; pass < passes; ++pass) {
    std::vector<uint16_t> filled = depth_mm;

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const size_t idx = static_cast<size_t>(y) * width + x;
        if (depth_mm[idx] != 0) {
          continue;
        }

        int count = 0;
        uint32_t sum = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          const int yy = y + dy;
          if (yy < 0 || yy >= height) {
            continue;
          }
          for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if ((dx == 0 && dy == 0) || xx < 0 || xx >= width) {
              continue;
            }
            const uint16_t neighbor = depth_mm[static_cast<size_t>(yy) * width + xx];
            if (neighbor == 0) {
              continue;
            }
            sum += neighbor;
            ++count;
          }
        }

        if (count > 0) {
          filled[idx] = static_cast<uint16_t>(std::lround(static_cast<double>(sum) / count));
        }
      }
    }

    depth_mm.swap(filled);
  }
}

void applySpatialFilter(std::vector<uint16_t>& depth_mm,
                        int width,
                        int height,
                        double alpha,
                        double delta_mm,
                        int hole_passes)
{
  if (hole_passes > 0) {
    applyHoleFillingFilter(depth_mm, width, height, hole_passes);
  }

  const double blend = clamp01(alpha);
  const double max_delta = std::max(0.0, delta_mm);
  std::vector<uint16_t> filtered = depth_mm;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = static_cast<size_t>(y) * width + x;
      const uint16_t center = depth_mm[idx];
      if (center == 0) {
        continue;
      }

      int count = 0;
      uint32_t sum = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= height) {
          continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx;
          if (xx < 0 || xx >= width) {
            continue;
          }
          const uint16_t neighbor = depth_mm[static_cast<size_t>(yy) * width + xx];
          if (neighbor == 0 ||
              std::abs(static_cast<int>(neighbor) - static_cast<int>(center)) > max_delta) {
            continue;
          }
          sum += neighbor;
          ++count;
        }
      }

      if (count > 1) {
        const double mean = static_cast<double>(sum) / count;
        filtered[idx] = static_cast<uint16_t>(
          std::lround((1.0 - blend) * center + blend * mean));
      }
    }
  }

  depth_mm.swap(filtered);
}

bool imageToDepthMm(const sensor_msgs::msg::Image& image,
                    std::vector<uint16_t>& depth_mm,
                    const rclcpp::Logger& logger)
{
  const auto encoding = image.encoding;
  const int width = static_cast<int>(image.width);
  const int height = static_cast<int>(image.height);

  if (width <= 0 || height <= 0) {
    RCLCPP_WARN_THROTTLE(logger, *rclcpp::Clock::make_shared(), 2000,
                         "Ignoring invalid depth image dimensions: %ux%u",
                         image.width, image.height);
    return false;
  }

  depth_mm.assign(static_cast<size_t>(width) * height, 0);

  if (encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
      encoding == sensor_msgs::image_encodings::MONO16) {
    const size_t min_step = static_cast<size_t>(width) * sizeof(uint16_t);
    if (image.step < min_step || image.data.size() < static_cast<size_t>(image.step) * height) {
      RCLCPP_WARN_THROTTLE(logger, *rclcpp::Clock::make_shared(), 2000,
                           "Ignoring malformed 16-bit depth image: step=%u width=%u height=%u bytes=%zu",
                           image.step, image.width, image.height, image.data.size());
      return false;
    }

    for (int y = 0; y < height; ++y) {
      const uint8_t* row = image.data.data() + static_cast<size_t>(y) * image.step;
      uint16_t* out = depth_mm.data() + static_cast<size_t>(y) * width;
      for (int x = 0; x < width; ++x) {
        const uint8_t* p = row + static_cast<size_t>(x) * sizeof(uint16_t);
        out[x] = image.is_bigendian
          ? static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1])
          : static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
      }
    }
    return true;
  }

  if (encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    const size_t min_step = static_cast<size_t>(width) * sizeof(float);
    if (image.step < min_step || image.data.size() < static_cast<size_t>(image.step) * height) {
      RCLCPP_WARN_THROTTLE(logger, *rclcpp::Clock::make_shared(), 2000,
                           "Ignoring malformed 32-bit depth image: step=%u width=%u height=%u bytes=%zu",
                           image.step, image.width, image.height, image.data.size());
      return false;
    }

    for (int y = 0; y < height; ++y) {
      const uint8_t* row = image.data.data() + static_cast<size_t>(y) * image.step;
      uint16_t* out = depth_mm.data() + static_cast<size_t>(y) * width;
      for (int x = 0; x < width; ++x) {
        const uint8_t* p = row + static_cast<size_t>(x) * sizeof(float);
        uint32_t bits = 0;
        if (image.is_bigendian) {
          bits = (static_cast<uint32_t>(p[0]) << 24) |
                 (static_cast<uint32_t>(p[1]) << 16) |
                 (static_cast<uint32_t>(p[2]) << 8) |
                 static_cast<uint32_t>(p[3]);
        } else {
          std::memcpy(&bits, p, sizeof(bits));
        }

        float meters = 0.0f;
        std::memcpy(&meters, &bits, sizeof(meters));
        if (!std::isfinite(meters) || meters <= 0.0f) {
          out[x] = 0;
          continue;
        }

        const double mm = std::round(static_cast<double>(meters) * 1000.0);
        out[x] = static_cast<uint16_t>(
          std::min<double>(mm, std::numeric_limits<uint16_t>::max()));
      }
    }
    return true;
  }

  RCLCPP_WARN_THROTTLE(logger, *rclcpp::Clock::make_shared(), 2000,
                       "Unsupported Orbbec depth encoding '%s'; expected 16UC1, mono16, or 32FC1",
                       encoding.c_str());
  return false;
}

struct Intrinsics
{
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

Intrinsics intrinsicsFromInfoOrFov(const std::optional<sensor_msgs::msg::CameraInfo>& info,
                                   int image_w,
                                   int image_h,
                                   double fallback_hfov_deg,
                                   double fallback_vfov_deg)
{
  if (info && info->k[0] > 0.0 && info->k[4] > 0.0 && info->width > 0 && info->height > 0) {
    const double sx = static_cast<double>(image_w) / static_cast<double>(info->width);
    const double sy = static_cast<double>(image_h) / static_cast<double>(info->height);
    return {
      info->k[0] * sx,
      info->k[4] * sy,
      info->k[2] * sx,
      info->k[5] * sy,
    };
  }

  const double hfov = fallback_hfov_deg * M_PI / 180.0;
  const double vfov = fallback_vfov_deg * M_PI / 180.0;
  return {
    static_cast<double>(image_w) / (2.0 * std::tan(hfov / 2.0)),
    static_cast<double>(image_h) / (2.0 * std::tan(vfov / 2.0)),
    static_cast<double>(image_w - 1) / 2.0,
    static_cast<double>(image_h - 1) / 2.0,
  };
}

}  // namespace

class OrbbecDepthBridge
{
public:
  explicit OrbbecDepthBridge(rclcpp::Node::SharedPtr node)
  : node_(std::move(node))
  {
    declare_and_get(node_, "input_depth_topic", input_depth_topic_);
    declare_and_get(node_, "input_depth_info_topic", input_depth_info_topic_);
    declare_and_get(node_, "topic_depth", topic_depth_);
    declare_and_get(node_, "topic_depth_info", topic_depth_info_);
    declare_and_get(node_, "topic_small", topic_small_);
    declare_and_get(node_, "topic_small_info", topic_small_info_);
    declare_and_get(node_, "frame_depth", frame_depth_);
    declare_and_get(node_, "out_w", out_w_);
    declare_and_get(node_, "out_h", out_h_);
    declare_and_get(node_, "hfov", hfov_);
    declare_and_get(node_, "vfov", vfov_);
    declare_and_get(node_, "min_dist", min_dist_);
    declare_and_get(node_, "max_dist", max_dist_);
    declare_and_get(node_, "publish_full", publish_full_);
    declare_and_get(node_, "publish_rate_hz", publish_rate_hz_);
    declare_and_get(node_, "use_input_frame_id", use_input_frame_id_);
    declare_and_get(node_, "enable_spatial_filter", enable_spatial_filter_);
    declare_and_get(node_, "spat_alpha", spat_alpha_);
    declare_and_get(node_, "spat_delta", spat_delta_);
    declare_and_get(node_, "spat_holes", spat_holes_);
    declare_and_get(node_, "enable_temporal_filter", enable_temporal_filter_);
    declare_and_get(node_, "temp_alpha", temp_alpha_);
    declare_and_get(node_, "temp_delta", temp_delta_);
    declare_and_get(node_, "enable_hole_filling_filter", enable_hole_filling_filter_);
    declare_and_get(node_, "hole1", hole1_);
    declare_and_get(node_, "hole2", hole2_);

    if (out_w_ <= 0 || out_h_ <= 0) {
      throw std::runtime_error("out_w and out_h must be positive");
    }
    if (min_dist_ < 0.0 || max_dist_ <= 0.0 || max_dist_ < min_dist_) {
      throw std::runtime_error("invalid min_dist/max_dist depth range");
    }
    if (spat_holes_ < 0 || hole1_ < 0 || hole2_ < 0) {
      throw std::runtime_error("hole filling parameters must be non-negative");
    }

    min_mm_ = static_cast<uint16_t>(std::lround(min_dist_ * 1000.0));
    max_mm_ = static_cast<uint16_t>(
      std::min<double>(std::lround(max_dist_ * 1000.0), std::numeric_limits<uint16_t>::max()));
    if (max_mm_ == 0) {
      max_mm_ = 1;
    }

    auto qos = rclcpp::SensorDataQoS();
    depth_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
      input_depth_topic_, qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { onDepth(std::move(msg)); });
    info_sub_ = node_->create_subscription<sensor_msgs::msg::CameraInfo>(
      input_depth_info_topic_, qos,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) { onInfo(std::move(msg)); });

    depth_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(topic_depth_, 1);
    depth_info_pub_ = node_->create_publisher<sensor_msgs::msg::CameraInfo>(topic_depth_info_, 1);
    small_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(topic_small_, 1);
    small_info_pub_ = node_->create_publisher<sensor_msgs::msg::CameraInfo>(topic_small_info_, 1);

    RCLCPP_INFO(node_->get_logger(),
                "Orbbec Gemini335 depth bridge: %s -> %s and %s (%dx%d), filters spatial=%s temporal=%s hole=%s",
                input_depth_topic_.c_str(), topic_depth_.c_str(), topic_small_.c_str(),
                out_w_, out_h_,
                enable_spatial_filter_ ? "on" : "off",
                enable_temporal_filter_ ? "on" : "off",
                enable_hole_filling_filter_ ? "on" : "off");
  }

private:
  void onInfo(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(info_mutex_);
    latest_info_ = *msg;
  }

  bool shouldPublishNow()
  {
    if (publish_rate_hz_ <= 0.0) {
      return true;
    }

    const auto now = node_->now();
    if (!last_publish_time_) {
      last_publish_time_ = now;
      return true;
    }

    const double elapsed = (now - *last_publish_time_).seconds();
    if (elapsed < 1.0 / publish_rate_hz_) {
      return false;
    }

    last_publish_time_ = now;
    return true;
  }

  void onDepth(sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    if (!shouldPublishNow()) {
      return;
    }

    const int width = static_cast<int>(msg->width);
    const int height = static_cast<int>(msg->height);
    std::vector<uint16_t> depth_mm;
    if (!imageToDepthMm(*msg, depth_mm, node_->get_logger())) {
      return;
    }
    applyDepthFilters(depth_mm, width, height);

    std::optional<sensor_msgs::msg::CameraInfo> info;
    {
      std::lock_guard<std::mutex> lock(info_mutex_);
      info = latest_info_;
    }

    const auto intr = intrinsicsFromInfoOrFov(info, width, height, hfov_, vfov_);
    const std::string frame_id = use_input_frame_id_ && !msg->header.frame_id.empty()
      ? msg->header.frame_id
      : frame_depth_;

    if (publish_full_) {
      publishFull(*msg, depth_mm, width, height, intr, frame_id);
    }
    publishSmall(*msg, depth_mm, width, height, intr, frame_id);
  }

  void applyDepthFilters(std::vector<uint16_t>& depth_mm, int width, int height)
  {
    if (enable_spatial_filter_) {
      applySpatialFilter(depth_mm, width, height, spat_alpha_, spat_delta_, spat_holes_);
    }
    if (enable_temporal_filter_) {
      applyTemporalFilter(depth_mm, width, height);
    }
    if (enable_hole_filling_filter_) {
      applyHoleFillingFilter(depth_mm, width, height, hole1_);
      applyHoleFillingFilter(depth_mm, width, height, hole2_);
    }
  }

  void applyTemporalFilter(std::vector<uint16_t>& depth_mm, int width, int height)
  {
    const size_t expected_size = static_cast<size_t>(width) * height;
    if (previous_depth_mm_.size() != expected_size) {
      previous_depth_mm_ = depth_mm;
      return;
    }

    const double blend = clamp01(temp_alpha_);
    const double max_delta = std::max(0.0, temp_delta_);
    std::vector<uint16_t> filtered = depth_mm;

    for (size_t i = 0; i < depth_mm.size(); ++i) {
      const uint16_t current = depth_mm[i];
      const uint16_t previous = previous_depth_mm_[i];
      if (current == 0 || previous == 0) {
        continue;
      }
      if (std::abs(static_cast<int>(current) - static_cast<int>(previous)) > max_delta) {
        continue;
      }

      filtered[i] = static_cast<uint16_t>(
        std::lround((1.0 - blend) * previous + blend * current));
    }

    previous_depth_mm_ = filtered;
    depth_mm.swap(filtered);
  }

  void publishFull(const sensor_msgs::msg::Image& input,
                   const std::vector<uint16_t>& depth_mm,
                   int width,
                   int height,
                   const Intrinsics& intr,
                   const std::string& frame_id)
  {
    std::vector<uint16_t> clamped(depth_mm.size());
    for (size_t i = 0; i < depth_mm.size(); ++i) {
      clamped[i] = clampDepthMm(depth_mm[i], min_mm_, max_mm_);
    }

    sensor_msgs::msg::Image out;
    out.header = input.header;
    out.header.frame_id = frame_id;
    out.width = static_cast<uint32_t>(width);
    out.height = static_cast<uint32_t>(height);
    out.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
    out.is_bigendian = false;
    out.step = static_cast<uint32_t>(width * sizeof(uint16_t));
    out.data.resize(static_cast<size_t>(out.step) * height);
    std::memcpy(out.data.data(), clamped.data(), out.data.size());
    depth_pub_->publish(out);

    sensor_msgs::msg::CameraInfo ci;
    ci.header = out.header;
    fillCameraInfo(ci, width, height, intr.fx, intr.fy, intr.cx, intr.cy);
    depth_info_pub_->publish(ci);
  }

  void publishSmall(const sensor_msgs::msg::Image& input,
                    const std::vector<uint16_t>& depth_mm,
                    int width,
                    int height,
                    const Intrinsics& intr,
                    const std::string& frame_id)
  {
    const double target_hfov = hfov_ * M_PI / 180.0;
    const double target_vfov = vfov_ * M_PI / 180.0;
    const double out_ratio = static_cast<double>(out_w_) / static_cast<double>(out_h_);

    int crop_w_from_h = static_cast<int>(
      std::round(2.0 * intr.fy * std::tan(target_vfov / 2.0) * out_ratio));
    int crop_h_from_h = static_cast<int>(
      std::round(2.0 * intr.fy * std::tan(target_vfov / 2.0)));
    int crop_w_from_w = static_cast<int>(
      std::round(2.0 * intr.fx * std::tan(target_hfov / 2.0)));
    int crop_h_from_w = static_cast<int>(std::round(crop_w_from_w / out_ratio));

    auto clamp_roi = [width, height](int& crop_w, int& crop_h) {
      crop_w = std::max(1, std::min(crop_w, width));
      crop_h = std::max(1, std::min(crop_h, height));
    };
    clamp_roi(crop_w_from_h, crop_h_from_h);
    clamp_roi(crop_w_from_w, crop_h_from_w);

    auto fov_error = [&intr, target_hfov, target_vfov](int crop_w, int crop_h) {
      const double hfov = 2.0 * std::atan(static_cast<double>(crop_w) / (2.0 * intr.fx));
      const double vfov = 2.0 * std::atan(static_cast<double>(crop_h) / (2.0 * intr.fy));
      return std::abs(hfov - target_hfov) + std::abs(vfov - target_vfov);
    };

    const bool use_from_h =
      fov_error(crop_w_from_h, crop_h_from_h) <= fov_error(crop_w_from_w, crop_h_from_w);
    const int crop_w = use_from_h ? crop_w_from_h : crop_w_from_w;
    const int crop_h = use_from_h ? crop_h_from_h : crop_h_from_w;
    const int x0 = (width - crop_w) / 2;
    const int y0 = (height - crop_h) / 2;

    std::vector<uint16_t> small(static_cast<size_t>(out_w_) * out_h_);
    const uint16_t* roi = depth_mm.data() + static_cast<size_t>(y0) * width + x0;
    resizeNearest16U(roi, crop_w, crop_h, width, small.data(), out_w_, out_h_);

    for (auto& depth : small) {
      depth = clampDepthMm(depth, min_mm_, max_mm_);
    }

    sensor_msgs::msg::Image out;
    out.header = input.header;
    out.header.frame_id = frame_id;
    out.width = static_cast<uint32_t>(out_w_);
    out.height = static_cast<uint32_t>(out_h_);
    out.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
    out.is_bigendian = false;
    out.step = static_cast<uint32_t>(out_w_ * sizeof(uint16_t));
    out.data.resize(static_cast<size_t>(out.step) * out_h_);
    std::memcpy(out.data.data(), small.data(), out.data.size());
    small_pub_->publish(out);

    double fx = intr.fx;
    double fy = intr.fy;
    double cx = intr.cx - x0;
    double cy = intr.cy - y0;
    fx *= static_cast<double>(out_w_) / static_cast<double>(crop_w);
    fy *= static_cast<double>(out_h_) / static_cast<double>(crop_h);
    cx *= static_cast<double>(out_w_) / static_cast<double>(crop_w);
    cy *= static_cast<double>(out_h_) / static_cast<double>(crop_h);

    sensor_msgs::msg::CameraInfo ci;
    ci.header = out.header;
    fillCameraInfo(ci, out_w_, out_h_, fx, fy, cx, cy);
    small_info_pub_->publish(ci);

    RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                         "Gemini335 bridge FOV(small): w=%d h=%d ROI=%dx%d@(%d,%d) fx=%.2f fy=%.2f",
                         out_w_, out_h_, crop_w, crop_h, x0, y0, fx, fy);
  }

  rclcpp::Node::SharedPtr node_;

  std::string input_depth_topic_ = "/orbbec/depth/image_raw";
  std::string input_depth_info_topic_ = "/orbbec/depth/camera_info";
  std::string topic_depth_ = "/camera/depth/image_raw";
  std::string topic_depth_info_ = "/camera/depth/camera_info";
  std::string topic_small_ = "/camera/depth/image_64x36";
  std::string topic_small_info_ = "/camera/depth/camera_info_64x36";
  std::string frame_depth_ = "camera_depth_optical_frame";
  int out_w_ = 64;
  int out_h_ = 36;
  double hfov_ = 89.24;
  double vfov_ = 58.06;
  double min_dist_ = 0.2;
  double max_dist_ = 2.5;
  bool publish_full_ = true;
  double publish_rate_hz_ = 60.0;
  bool use_input_frame_id_ = false;
  bool enable_spatial_filter_ = true;
  double spat_alpha_ = 0.45;
  double spat_delta_ = 20.0;
  int spat_holes_ = 2;
  bool enable_temporal_filter_ = true;
  double temp_alpha_ = 0.45;
  double temp_delta_ = 20.0;
  bool enable_hole_filling_filter_ = true;
  int hole1_ = 1;
  int hole2_ = 2;
  uint16_t min_mm_ = 200;
  uint16_t max_mm_ = 2500;

  std::mutex info_mutex_;
  std::optional<sensor_msgs::msg::CameraInfo> latest_info_;
  std::optional<rclcpp::Time> last_publish_time_;
  std::vector<uint16_t> previous_depth_mm_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr small_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr small_info_pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("orbbec_depth_bridge");

  try {
    auto bridge = std::make_shared<OrbbecDepthBridge>(node);
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_FATAL(node->get_logger(), "Failed to start Orbbec depth bridge: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
