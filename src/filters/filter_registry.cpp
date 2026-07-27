#include "lmp/filters/filter_registry.hpp"

#include "lmp/filters/auto_frame_filter.hpp"
#include "lmp/filters/background_blur_filter.hpp"
#include "lmp/filters/box_blur_filter.hpp"
#include "lmp/filters/brightness_filter.hpp"
#include "lmp/filters/color_adjust_filter.hpp"
#include "lmp/filters/contrast_filter.hpp"
#include "lmp/filters/exposure_filter.hpp"
#include "lmp/filters/fps_overlay_filter.hpp"
#include "lmp/filters/gamma_filter.hpp"
#include "lmp/filters/gaussian_blur_filter.hpp"
#include "lmp/filters/grayscale_filter.hpp"
#include "lmp/filters/histogram_filter.hpp"
#include "lmp/filters/identity_filter.hpp"
#include "lmp/filters/negative_filter.hpp"
#include "lmp/filters/saturation_filter.hpp"
#include "lmp/filters/sepia_filter.hpp"
#include "lmp/filters/sharpen_filter.hpp"
#include "lmp/filters/sobel_filter.hpp"
#include "lmp/filters/temperature_filter.hpp"
#include "lmp/filters/text_overlay_filter.hpp"
#include "lmp/filters/timestamp_overlay_filter.hpp"
#include "lmp/filters/tint_filter.hpp"
#include "lmp/filters/white_balance_filter.hpp"

#include "filter_config_params.hpp"

#include <stdexcept>

namespace lmp::filters {

void FilterRegistry::register_filter(std::string type, FilterFactory factory) {
  if (type.empty()) {
    throw std::invalid_argument("filter type cannot be empty");
  }
  if (!factory) {
    throw std::invalid_argument("filter factory cannot be empty");
  }
  const auto [_, inserted] =
      factories_.emplace(std::move(type), std::move(factory));
  if (!inserted) {
    throw std::invalid_argument("filter type is already registered");
  }
}

std::unique_ptr<IVideoFilter>
FilterRegistry::create(const config::FilterConfig &config) const {
  const auto found = factories_.find(config.type);
  if (found == factories_.end()) {
    throw std::invalid_argument("unknown filter type: " + config.type);
  }
  return found->second(config);
}

bool FilterRegistry::contains(std::string_view type) const noexcept {
  return factories_.find(std::string(type)) != factories_.end();
}

std::size_t FilterRegistry::size() const noexcept { return factories_.size(); }

FilterRegistry create_default_registry() {
  FilterRegistry registry;
  registry.register_filter("identity", [](const config::FilterConfig &config) {
    static_cast<void>(config);
    return std::make_unique<IdentityFilter>();
  });
  registry.register_filter("grayscale", [](const config::FilterConfig &config) {
    static_cast<void>(config);
    return std::make_unique<GrayscaleFilter>();
  });
  registry.register_filter("negative", [](const config::FilterConfig &config) {
    static_cast<void>(config);
    return std::make_unique<NegativeFilter>();
  });
  registry.register_filter("sepia", [](const config::FilterConfig &config) {
    static_cast<void>(config);
    return std::make_unique<SepiaFilter>();
  });
  registry.register_filter("box_blur", [](const config::FilterConfig &config) {
    return std::make_unique<BoxBlurFilter>(
        detail::radius_parameter(config, 1U));
  });
  registry.register_filter("blur", [](const config::FilterConfig &config) {
    return std::make_unique<BoxBlurFilter>(
        detail::radius_parameter(config, 1U));
  });
  registry.register_filter("gaussian_blur",
                           [](const config::FilterConfig &config) {
                             return std::make_unique<GaussianBlurFilter>(
                                 detail::radius_parameter(config, 1U));
                           });
  registry.register_filter("sharpen", [](const config::FilterConfig &config) {
    const auto amount = detail::double_parameter(config, "amount", 1.0);
    return std::make_unique<SharpenFilter>(amount);
  });
  registry.register_filter("sobel", [](const config::FilterConfig &config) {
    static_cast<void>(config);
    return std::make_unique<SobelFilter>();
  });
  registry.register_filter("gamma", [](const config::FilterConfig &config) {
    return std::make_unique<GammaFilter>(
        detail::double_parameter(config, "gamma", 1.0));
  });
  registry.register_filter("exposure", [](const config::FilterConfig &config) {
    return std::make_unique<ExposureFilter>(
        detail::double_parameter(config, "stops", 0.0));
  });
  registry.register_filter("contrast", [](const config::FilterConfig &config) {
    return std::make_unique<ContrastFilter>(
        detail::double_parameter(config, "factor", 1.0));
  });
  registry.register_filter(
      "brightness", [](const config::FilterConfig &config) {
        return std::make_unique<BrightnessFilter>(
            detail::double_parameter(config, "offset", 0.0));
      });
  registry.register_filter(
      "saturation", [](const config::FilterConfig &config) {
        return std::make_unique<SaturationFilter>(
            detail::double_parameter(config, "factor", 1.0));
      });
  registry.register_filter(
      "color_adjust", [](const config::FilterConfig &config) {
        return std::make_unique<ColorAdjustFilter>(
            detail::double_parameter(config, "brightness", 0.0),
            detail::double_parameter(config, "contrast", 1.0),
            detail::double_parameter(config, "saturation", 1.0),
            detail::string_parameter(config, "backend", "cpu"));
      });
  registry.register_filter(
      "white_balance", [](const config::FilterConfig &config) {
        return std::make_unique<WhiteBalanceFilter>(
            detail::double_parameter(config, "red_gain", 1.0),
            detail::double_parameter(config, "green_gain", 1.0),
            detail::double_parameter(config, "blue_gain", 1.0));
      });
  registry.register_filter(
      "temperature", [](const config::FilterConfig &config) {
        return std::make_unique<TemperatureFilter>(
            detail::double_parameter(config, "amount", 0.0));
      });
  registry.register_filter("tint", [](const config::FilterConfig &config) {
    return std::make_unique<TintFilter>(
        detail::double_parameter(config, "amount", 0.0));
  });
  registry.register_filter(
      "text_overlay", [](const config::FilterConfig &config) {
        return std::make_unique<TextOverlayFilter>(
            detail::string_parameter(config, "text", "LMP"),
            detail::coordinate_parameter(config, "x", 0U),
            detail::coordinate_parameter(config, "y", 0U));
      });
  registry.register_filter("text", [](const config::FilterConfig &config) {
    return std::make_unique<TextOverlayFilter>(
        detail::string_parameter(config, "text", "LMP"),
        detail::coordinate_parameter(config, "x", 0U),
        detail::coordinate_parameter(config, "y", 0U));
  });
  registry.register_filter("fps_overlay",
                           [](const config::FilterConfig &config) {
                             return std::make_unique<FpsOverlayFilter>(
                                 detail::double_parameter(config, "fps", 0.0),
                                 detail::coordinate_parameter(config, "x", 0U),
                                 detail::coordinate_parameter(config, "y", 0U));
                           });
  registry.register_filter("fps", [](const config::FilterConfig &config) {
    return std::make_unique<FpsOverlayFilter>(
        detail::double_parameter(config, "fps", 0.0),
        detail::coordinate_parameter(config, "x", 0U),
        detail::coordinate_parameter(config, "y", 0U));
  });
  registry.register_filter("timestamp_overlay",
                           [](const config::FilterConfig &config) {
                             return std::make_unique<TimestampOverlayFilter>(
                                 detail::coordinate_parameter(config, "x", 0U),
                                 detail::coordinate_parameter(config, "y", 0U));
                           });
  registry.register_filter("timestamp", [](const config::FilterConfig &config) {
    return std::make_unique<TimestampOverlayFilter>(
        detail::coordinate_parameter(config, "x", 0U),
        detail::coordinate_parameter(config, "y", 0U));
  });
  registry.register_filter("histogram", [](const config::FilterConfig &config) {
    return std::make_unique<HistogramFilter>(
        detail::bool_parameter(config, "draw_overlay", false),
        detail::coordinate_parameter(config, "x", 0U),
        detail::coordinate_parameter(config, "y", 0U),
        detail::coordinate_parameter(config, "width", 64U),
        detail::coordinate_parameter(config, "height", 32U));
  });
  registry.register_filter(
      "background_blur", [](const config::FilterConfig &config) {
        return std::make_unique<BackgroundBlurFilter>(
            detail::radius_parameter(config, 1U),
            static_cast<std::uint8_t>(
                detail::integer_parameter(config, "foreground_threshold", 128)),
            detail::string_parameter(config, "backend", "cpu"),
            detail::double_parameter(config, "brightness", 0.0),
            detail::double_parameter(config, "contrast", 1.0),
            detail::double_parameter(config, "saturation", 1.0),
            detail::bool_parameter(config, "auto_frame", false),
            detail::double_parameter(config, "target_fill", 0.62),
            detail::double_parameter(config, "max_zoom", 1.8),
            detail::string_parameter(config, "mask_mode", "luminance"),
            detail::double_parameter(config, "mask_width", 0.28),
            detail::double_parameter(config, "mask_height", 0.42),
            detail::string_parameter(config, "model_path",
                                     "assets/models/person-segmentation.onnx"),
            detail::coordinate_parameter(config, "inference_interval", 3U),
            detail::double_parameter(config, "mask_smoothing", 0.70),
            detail::string_parameter(config, "fallback_mask_mode",
                                     "tracked_center"),
            detail::string_parameter(config, "input_shape", ""),
            detail::string_parameter(config, "output_shape", ""),
            detail::string_parameter(config, "provider", "auto"),
            detail::bool_parameter(config, "allow_provider_fallback", true),
            detail::string_parameter(config, "openvino_device", "CPU"),
            detail::coordinate_parameter(config, "mask_expand", 1U),
            detail::coordinate_parameter(config, "mask_feather", 3U),
            detail::bool_parameter(config, "invert_mask", false),
            detail::bool_parameter(config, "keep_largest_component", false),
            detail::double_parameter(config, "min_mask_coverage", 0.02),
            detail::double_parameter(config, "max_mask_coverage", 0.85),
            detail::double_parameter(config, "hint_y_offset", 0.0),
            detail::string_parameter(config, "background_mode", "blur"),
            detail::string_parameter(config, "background_path", ""),
            detail::string_parameter(config, "background_color", "#1b1f2a"));
      });
  registry.register_filter(
      "auto_frame", [](const config::FilterConfig &config) {
        return std::make_unique<AutoFrameFilter>(
            detail::double_parameter(config, "target_fill", 0.62),
            detail::double_parameter(config, "max_zoom", 1.8),
            static_cast<std::uint8_t>(
                detail::integer_parameter(config, "foreground_threshold", 128)),
            detail::double_parameter(config, "smoothing", 0.82),
            detail::double_parameter(config, "dead_zone", 0.04));
      });
  return registry;
}

} // namespace lmp::filters
