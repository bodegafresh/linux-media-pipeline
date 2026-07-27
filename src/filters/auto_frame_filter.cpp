#include "lmp/filters/auto_frame_filter.hpp"

#include "spatial_filter.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace lmp::filters {
namespace {

struct Bounds {
  std::uint32_t min_x;
  std::uint32_t min_y;
  std::uint32_t max_x;
  std::uint32_t max_y;
};

std::uint8_t luminance(detail::RgbPixel pixel) {
  return detail::clamp_to_byte((0.299 * pixel.red) + (0.587 * pixel.green) +
                               (0.114 * pixel.blue));
}

std::optional<Bounds>
foreground_bounds(const std::vector<detail::RgbPixel> &source,
                  std::uint32_t width, std::uint32_t height,
                  std::uint8_t threshold) {
  auto bounds = Bounds{width, height, 0U, 0U};
  bool found = false;
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const auto pixel = source[detail::pixel_index(x, y, width)];
      if (luminance(pixel) < threshold) {
        continue;
      }
      bounds.min_x = std::min(bounds.min_x, x);
      bounds.min_y = std::min(bounds.min_y, y);
      bounds.max_x = std::max(bounds.max_x, x);
      bounds.max_y = std::max(bounds.max_y, y);
      found = true;
    }
  }
  if (!found) {
    return std::nullopt;
  }
  return bounds;
}

std::uint32_t rounded_odd_crop(double value, std::uint32_t upper) {
  const auto rounded = static_cast<std::uint32_t>(std::ceil(value));
  return std::clamp(rounded, 1U, upper);
}

} // namespace

AutoFrameFilter::AutoFrameFilter(double target_fill, double max_zoom,
                                 std::uint8_t foreground_threshold)
    : AutoFrameFilter(target_fill, max_zoom, foreground_threshold, 0.82, 0.04) {
}

AutoFrameFilter::AutoFrameFilter(double target_fill, double max_zoom,
                                 std::uint8_t foreground_threshold,
                                 double smoothing, double dead_zone)
    : target_fill_(target_fill), max_zoom_(max_zoom),
      foreground_threshold_(foreground_threshold), smoothing_(smoothing),
      dead_zone_(dead_zone) {
  if (target_fill_ <= 0.0 || target_fill_ > 1.0) {
    throw std::invalid_argument("auto_frame target_fill must be in (0, 1]");
  }
  if (max_zoom_ < 1.0) {
    throw std::invalid_argument("auto_frame max_zoom must be >= 1");
  }
  if (smoothing_ < 0.0 || smoothing_ >= 1.0) {
    throw std::invalid_argument("auto_frame smoothing must be in [0, 1)");
  }
  if (dead_zone_ < 0.0 || dead_zone_ > 1.0) {
    throw std::invalid_argument("auto_frame dead_zone must be in [0, 1]");
  }
}

AutoFrameFilter::Crop AutoFrameFilter::smooth_crop(Crop desired) const {
  if (!previous_crop_.has_value()) {
    previous_crop_ = desired;
    return desired;
  }

  const auto previous_center_x =
      previous_crop_->x + (previous_crop_->width / 2.0);
  const auto previous_center_y =
      previous_crop_->y + (previous_crop_->height / 2.0);
  const auto desired_center_x = desired.x + (desired.width / 2.0);
  const auto desired_center_y = desired.y + (desired.height / 2.0);
  const auto movement = std::hypot(desired_center_x - previous_center_x,
                                   desired_center_y - previous_center_y);
  const auto dead_zone_pixels =
      dead_zone_ * std::min(desired.width, desired.height);
  if (movement <= dead_zone_pixels) {
    return *previous_crop_;
  }

  const auto blend = 1.0 - smoothing_;
  auto smoothed = Crop{
      previous_crop_->x + ((desired.x - previous_crop_->x) * blend),
      previous_crop_->y + ((desired.y - previous_crop_->y) * blend),
      previous_crop_->width + ((desired.width - previous_crop_->width) * blend),
      previous_crop_->height +
          ((desired.height - previous_crop_->height) * blend)};
  previous_crop_ = smoothed;
  return smoothed;
}

void AutoFrameFilter::process(frame::Frame &frame) const {
  const auto source = detail::read_packed_rgb(frame);
  const auto bounds = foreground_bounds(source, frame.width(), frame.height(),
                                        foreground_threshold_);
  if (!bounds.has_value()) {
    frame.metadata()["auto_frame"] = "not_found";
    return;
  }

  const auto frame_width = frame.width();
  const auto frame_height = frame.height();
  const auto aspect =
      static_cast<double>(frame_width) / static_cast<double>(frame_height);
  const auto person_width =
      static_cast<double>(bounds->max_x - bounds->min_x + 1U);
  const auto person_height =
      static_cast<double>(bounds->max_y - bounds->min_y + 1U);

  auto crop_width = std::max(person_width / target_fill_,
                             (person_height / target_fill_) * aspect);
  crop_width =
      std::max(crop_width, static_cast<double>(frame_width) / max_zoom_);
  crop_width = std::min(crop_width, static_cast<double>(frame_width));
  auto crop_height = crop_width / aspect;
  if (crop_height > static_cast<double>(frame_height)) {
    crop_height = static_cast<double>(frame_height);
    crop_width = crop_height * aspect;
  }

  auto crop_w = rounded_odd_crop(crop_width, frame_width);
  auto crop_h = rounded_odd_crop(crop_height, frame_height);
  const auto center_x = (static_cast<double>(bounds->min_x) +
                         static_cast<double>(bounds->max_x)) /
                        2.0;
  const auto center_y = (static_cast<double>(bounds->min_y) +
                         static_cast<double>(bounds->max_y)) /
                        2.0;
  const auto max_crop_x = frame_width - crop_w;
  const auto max_crop_y = frame_height - crop_h;
  auto crop_x = static_cast<std::uint32_t>(
      std::clamp(static_cast<int>(std::round(
                     center_x - (static_cast<double>(crop_w) / 2.0))),
                 0, static_cast<int>(max_crop_x)));
  auto crop_y = static_cast<std::uint32_t>(
      std::clamp(static_cast<int>(std::round(
                     center_y - (static_cast<double>(crop_h) / 2.0))),
                 0, static_cast<int>(max_crop_y)));

  const auto smoothed = smooth_crop(
      Crop{static_cast<double>(crop_x), static_cast<double>(crop_y),
           static_cast<double>(crop_w), static_cast<double>(crop_h)});
  crop_w = rounded_odd_crop(smoothed.width, frame_width);
  crop_h = rounded_odd_crop(smoothed.height, frame_height);
  crop_x = static_cast<std::uint32_t>(
      std::clamp(static_cast<int>(std::round(smoothed.x)), 0,
                 static_cast<int>(frame_width - crop_w)));
  crop_y = static_cast<std::uint32_t>(
      std::clamp(static_cast<int>(std::round(smoothed.y)), 0,
                 static_cast<int>(frame_height - crop_h)));

  std::vector<detail::RgbPixel> output(source.size());
  for (std::uint32_t y = 0; y < frame_height; ++y) {
    const auto source_y = static_cast<std::uint32_t>(
        crop_y + ((static_cast<std::uint64_t>(y) * crop_h) / frame_height));
    for (std::uint32_t x = 0; x < frame_width; ++x) {
      const auto source_x = static_cast<std::uint32_t>(
          crop_x + ((static_cast<std::uint64_t>(x) * crop_w) / frame_width));
      output[detail::pixel_index(x, y, frame_width)] =
          source[detail::pixel_index(source_x, source_y, frame_width)];
    }
  }

  detail::write_packed_rgb(frame, output);

  std::ostringstream metadata;
  metadata << crop_x << ',' << crop_y << ',' << crop_w << ',' << crop_h;
  frame.metadata()["auto_frame"] = metadata.str();
}

std::string_view AutoFrameFilter::type() const noexcept { return "auto_frame"; }

} // namespace lmp::filters
