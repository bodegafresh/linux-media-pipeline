#pragma once

#include "lmp/ai/onnx_runtime_engine.hpp"
#include "lmp/ai/segmentation_mask.hpp"
#include "lmp/decoder/ffmpeg_decoder.hpp"
#include "lmp/filters/video_filter.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lmp::filters {

class BackgroundBlurFilter final : public IVideoFilter {
public:
  BackgroundBlurFilter(std::uint32_t radius, std::uint8_t foreground_threshold);
  BackgroundBlurFilter(std::uint32_t radius, std::uint8_t foreground_threshold,
                       std::string backend);
  BackgroundBlurFilter(std::uint32_t radius, std::uint8_t foreground_threshold,
                       std::string backend, double brightness, double contrast,
                       double saturation);
  BackgroundBlurFilter(std::uint32_t radius, std::uint8_t foreground_threshold,
                       std::string backend, double brightness, double contrast,
                       double saturation, bool auto_frame, double target_fill,
                       double max_zoom, std::string mask_mode,
                       double mask_width, double mask_height,
                       std::string model_path, std::uint32_t inference_interval,
                       double mask_smoothing, std::string fallback_mask_mode,
                       std::string input_shape, std::string output_shape,
                       std::string requested_provider,
                       bool allow_provider_fallback,
                       std::string openvino_device, std::uint32_t mask_expand,
                       std::uint32_t mask_feather, bool invert_mask,
                       bool keep_largest_component, double min_mask_coverage,
                       double max_mask_coverage, double hint_y_offset,
                       std::string background_mode = "blur",
                       std::string background_path = "",
                       std::string background_color = "#1b1f2a");

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;

private:
  void process_cpu(frame::Frame &frame) const;
  [[nodiscard]] bool process_opencl(frame::Frame &frame) const;
  [[nodiscard]] std::optional<ai::SegmentationMask>
  person_mask(frame::Frame &frame) const;
  [[nodiscard]] std::vector<std::uint8_t>
  background_pixels(const frame::Frame &frame) const;

  std::uint32_t radius_;
  std::uint8_t foreground_threshold_;
  std::string backend_;
  double brightness_;
  double contrast_;
  double saturation_;
  bool auto_frame_;
  double target_fill_;
  double max_zoom_;
  std::string mask_mode_;
  double mask_width_;
  double mask_height_;
  std::string model_path_;
  std::uint32_t inference_interval_;
  double mask_smoothing_;
  std::string fallback_mask_mode_;
  std::string input_shape_;
  std::string output_shape_;
  std::string requested_provider_;
  bool allow_provider_fallback_;
  std::string openvino_device_;
  std::uint32_t mask_expand_;
  std::uint32_t mask_feather_;
  bool invert_mask_;
  bool keep_largest_component_;
  double min_mask_coverage_;
  double max_mask_coverage_;
  std::string background_mode_;
  std::string background_path_;
  std::string background_color_;
  mutable std::unique_ptr<lmp::decoder::FfmpegDecoder> background_decoder_;
  mutable std::vector<std::uint8_t> static_background_;
  mutable std::string background_error_;
  mutable std::unique_ptr<ai::OnnxRuntimeEngine> onnx_engine_;
  mutable std::optional<ai::SegmentationMask> last_good_person_mask_;
  mutable std::uint32_t last_good_person_mask_reuse_count_ = 0;
  mutable std::uint32_t bad_person_mask_streak_ = 0;
  mutable std::uint32_t onnx_mask_cooldown_frames_ = 0;
  mutable std::optional<std::array<double, 4>> previous_auto_frame_crop_;
  mutable bool onnx_error_reported_;
};

} // namespace lmp::filters
