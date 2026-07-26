#include "lmp/config/config_loader.hpp"
#include "lmp/filters/box_blur_filter.hpp"
#include "lmp/filters/brightness_filter.hpp"
#include "lmp/filters/contrast_filter.hpp"
#include "lmp/filters/exposure_filter.hpp"
#include "lmp/filters/filter_pipeline.hpp"
#include "lmp/filters/filter_registry.hpp"
#include "lmp/filters/gamma_filter.hpp"
#include "lmp/filters/gaussian_blur_filter.hpp"
#include "lmp/filters/grayscale_filter.hpp"
#include "lmp/filters/negative_filter.hpp"
#include "lmp/filters/saturation_filter.hpp"
#include "lmp/filters/sepia_filter.hpp"
#include "lmp/filters/sharpen_filter.hpp"
#include "lmp/filters/sobel_filter.hpp"
#include "lmp/filters/temperature_filter.hpp"
#include "lmp/filters/tint_filter.hpp"
#include "lmp/filters/white_balance_filter.hpp"
#include "lmp/frame/frame.hpp"
#include "lmp/version.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

bool expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    return false;
  }
  return true;
}

lmp::frame::Frame make_rgba_frame(std::vector<std::uint8_t> data) {
  return lmp::frame::Frame{2U,
                           1U,
                           lmp::frame::PixelFormat::Rgba,
                           std::move(data),
                           std::vector<std::size_t>{8U},
                           lmp::frame::Frame::Clock::now()};
}

lmp::frame::Frame make_rgb_frame(std::uint32_t width, std::uint32_t height,
                                 std::vector<std::uint8_t> data) {
  return lmp::frame::Frame{
      width,
      height,
      lmp::frame::PixelFormat::Rgb,
      std::move(data),
      std::vector<std::size_t>{static_cast<std::size_t>(width) * 3U},
      lmp::frame::Frame::Clock::now()};
}

std::vector<std::uint8_t> bytes(const lmp::frame::Frame &frame) {
  return {frame.data().begin(), frame.data().end()};
}

} // namespace

int main() {
  const auto current = lmp::version();
  bool ok = true;
  ok = expect(current.major == 0, "major version") && ok;
  ok = expect(current.minor == 1, "minor version") && ok;
  ok = expect(current.patch == 0, "patch version") && ok;
  ok = expect(lmp::version_string() == "0.1.0", "version string") && ok;

  const lmp::config::ConfigLoader loader;
  const auto config = loader.load_file("config/default.yaml");
  ok = expect(config.capture.type == "gopro_udp", "capture type") && ok;
  ok = expect(config.capture.address == "udp://0.0.0.0:8554",
              "capture address") &&
       ok;
  ok = expect(config.gpu.backend == "opencl", "gpu backend") && ok;
  ok = expect(config.pipeline.threads == "auto", "pipeline threads") && ok;
  ok = expect(config.pipeline.queue_size == 4U, "pipeline queue size") && ok;
  ok = expect(config.filters.size() == 16U, "filter count") && ok;
  ok = expect(config.filters.front().type == "identity",
              "identity filter config") &&
       ok;
  ok = expect(config.filters.front().enabled, "identity filter enabled") && ok;
  ok = expect(config.output.type == "v4l2", "output type") && ok;
  ok = expect(config.output.device == "/dev/video20", "output device") && ok;

  auto frame = lmp::frame::Frame{
      2U,
      2U,
      lmp::frame::PixelFormat::Rgba,
      std::vector<std::uint8_t>{1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U,
                                12U, 13U, 14U, 15U, 16U},
      std::vector<std::size_t>{8U},
      lmp::frame::Frame::Clock::now(),
      {{"source", "test"}}};

  const auto before =
      std::vector<std::uint8_t>{frame.data().begin(), frame.data().end()};
  const auto registry = lmp::filters::create_default_registry();
  const auto pipeline =
      lmp::filters::FilterPipeline::from_config(config.filters, registry);
  pipeline.process(frame);

  const auto after =
      std::vector<std::uint8_t>{frame.data().begin(), frame.data().end()};
  ok = expect(registry.contains("identity"), "identity registered") && ok;
  ok = expect(registry.contains("grayscale"), "grayscale registered") && ok;
  ok = expect(registry.contains("negative"), "negative registered") && ok;
  ok = expect(registry.contains("sepia"), "sepia registered") && ok;
  ok = expect(registry.contains("box_blur"), "box blur registered") && ok;
  ok = expect(registry.contains("blur"), "blur alias registered") && ok;
  ok = expect(registry.contains("gaussian_blur"), "gaussian blur registered") &&
       ok;
  ok = expect(registry.contains("sharpen"), "sharpen registered") && ok;
  ok = expect(registry.contains("sobel"), "sobel registered") && ok;
  ok = expect(registry.contains("gamma"), "gamma registered") && ok;
  ok = expect(registry.contains("exposure"), "exposure registered") && ok;
  ok = expect(registry.contains("contrast"), "contrast registered") && ok;
  ok = expect(registry.contains("brightness"), "brightness registered") && ok;
  ok = expect(registry.contains("saturation"), "saturation registered") && ok;
  ok = expect(registry.contains("white_balance"), "white balance registered") &&
       ok;
  ok = expect(registry.contains("temperature"), "temperature registered") && ok;
  ok = expect(registry.contains("tint"), "tint registered") && ok;
  ok = expect(registry.size() == 17U, "default registry size") && ok;
  ok = expect(pipeline.size() == 1U, "identity pipeline size") && ok;
  ok = expect(before == after, "identity keeps frame bytes unchanged") && ok;
  ok = expect(frame.metadata().at("source") == "test", "frame metadata") && ok;

  auto grayscale = make_rgba_frame({10U, 20U, 30U, 40U, 200U, 100U, 50U, 255U});
  lmp::filters::GrayscaleFilter{}.process(grayscale);
  ok = expect(bytes(grayscale) == std::vector<std::uint8_t>{18U, 18U, 18U, 40U,
                                                            124U, 124U, 124U,
                                                            255U},
              "grayscale rgba") &&
       ok;

  auto negative = make_rgba_frame({10U, 20U, 30U, 40U, 200U, 100U, 50U, 255U});
  lmp::filters::NegativeFilter{}.process(negative);
  ok = expect(bytes(negative) == std::vector<std::uint8_t>{245U, 235U, 225U,
                                                           40U, 55U, 155U, 205U,
                                                           255U},
              "negative rgba") &&
       ok;

  auto sepia = make_rgba_frame({10U, 20U, 30U, 40U, 200U, 100U, 50U, 255U});
  lmp::filters::SepiaFilter{}.process(sepia);
  ok = expect(bytes(sepia) == std::vector<std::uint8_t>{25U, 22U, 17U, 40U,
                                                        165U, 147U, 114U, 255U},
              "sepia rgba") &&
       ok;

  auto bgr = lmp::frame::Frame{1U,
                               1U,
                               lmp::frame::PixelFormat::Bgr,
                               std::vector<std::uint8_t>{30U, 20U, 10U},
                               std::vector<std::size_t>{3U},
                               lmp::frame::Frame::Clock::now()};
  lmp::filters::NegativeFilter{}.process(bgr);
  ok = expect(bytes(bgr) == std::vector<std::uint8_t>{225U, 235U, 245U},
              "negative bgr channel order") &&
       ok;

  auto box_blur =
      make_rgb_frame(3U, 1U, {0U, 0U, 0U, 90U, 90U, 90U, 180U, 180U, 180U});
  lmp::filters::BoxBlurFilter{1U}.process(box_blur);
  ok = expect(bytes(box_blur) == std::vector<std::uint8_t>{30U, 30U, 30U, 90U,
                                                           90U, 90U, 150U, 150U,
                                                           150U},
              "box blur rgb") &&
       ok;

  auto gaussian_blur =
      make_rgb_frame(3U, 1U, {0U, 0U, 0U, 90U, 90U, 90U, 180U, 180U, 180U});
  lmp::filters::GaussianBlurFilter{1U}.process(gaussian_blur);
  ok = expect(bytes(gaussian_blur) ==
                  std::vector<std::uint8_t>{23U, 23U, 23U, 90U, 90U, 90U, 158U,
                                            158U, 158U},
              "gaussian blur rgb") &&
       ok;

  auto sharpen = make_rgb_frame(
      3U, 1U, {50U, 50U, 50U, 100U, 100U, 100U, 150U, 150U, 150U});
  lmp::filters::SharpenFilter{1.0}.process(sharpen);
  ok = expect(bytes(sharpen) == std::vector<std::uint8_t>{0U, 0U, 0U, 100U,
                                                          100U, 100U, 200U,
                                                          200U, 200U},
              "sharpen rgb") &&
       ok;

  auto sobel =
      make_rgb_frame(3U, 3U, {0U, 0U, 0U, 0U, 0U, 0U, 255U, 255U, 255U,
                              0U, 0U, 0U, 0U, 0U, 0U, 255U, 255U, 255U,
                              0U, 0U, 0U, 0U, 0U, 0U, 255U, 255U, 255U});
  lmp::filters::SobelFilter{}.process(sobel);
  ok = expect(bytes(sobel) ==
                  std::vector<std::uint8_t>{
                      0U, 0U, 0U, 255U, 255U, 255U, 255U, 255U, 255U,
                      0U, 0U, 0U, 255U, 255U, 255U, 255U, 255U, 255U,
                      0U, 0U, 0U, 255U, 255U, 255U, 255U, 255U, 255U},
              "sobel rgb") &&
       ok;

  auto brightness =
      make_rgba_frame({10U, 100U, 240U, 255U, 0U, 20U, 40U, 128U});
  lmp::filters::BrightnessFilter{20.0}.process(brightness);
  ok = expect(bytes(brightness) == std::vector<std::uint8_t>{30U, 120U, 255U,
                                                             255U, 20U, 40U,
                                                             60U, 128U},
              "brightness rgba") &&
       ok;

  auto contrast =
      make_rgba_frame({100U, 128U, 160U, 255U, 50U, 200U, 250U, 128U});
  lmp::filters::ContrastFilter{2.0}.process(contrast);
  ok =
      expect(bytes(contrast) == std::vector<std::uint8_t>{72U, 128U, 192U, 255U,
                                                          0U, 255U, 255U, 128U},
             "contrast rgba") &&
      ok;

  auto exposure = make_rgba_frame({10U, 100U, 200U, 255U, 20U, 40U, 80U, 128U});
  lmp::filters::ExposureFilter{1.0}.process(exposure);
  ok =
      expect(bytes(exposure) == std::vector<std::uint8_t>{20U, 200U, 255U, 255U,
                                                          40U, 80U, 160U, 128U},
             "exposure rgba") &&
      ok;

  auto gamma = make_rgba_frame({64U, 128U, 255U, 255U, 16U, 81U, 144U, 128U});
  lmp::filters::GammaFilter{2.0}.process(gamma);
  ok = expect(bytes(gamma) == std::vector<std::uint8_t>{128U, 181U, 255U, 255U,
                                                        64U, 144U, 192U, 128U},
              "gamma rgba") &&
       ok;

  auto saturation =
      make_rgba_frame({100U, 150U, 200U, 255U, 10U, 20U, 30U, 128U});
  lmp::filters::SaturationFilter{0.0}.process(saturation);
  ok = expect(bytes(saturation) == std::vector<std::uint8_t>{141U, 141U, 141U,
                                                             255U, 18U, 18U,
                                                             18U, 128U},
              "saturation rgba") &&
       ok;

  auto white_balance =
      make_rgba_frame({100U, 100U, 100U, 255U, 200U, 50U, 25U, 128U});
  lmp::filters::WhiteBalanceFilter{1.2, 1.0, 0.5}.process(white_balance);
  ok = expect(bytes(white_balance) == std::vector<std::uint8_t>{120U, 100U, 50U,
                                                                255U, 240U, 50U,
                                                                13U, 128U},
              "white balance rgba") &&
       ok;

  auto temperature =
      make_rgba_frame({100U, 100U, 100U, 255U, 250U, 20U, 5U, 128U});
  lmp::filters::TemperatureFilter{20.0}.process(temperature);
  ok = expect(bytes(temperature) == std::vector<std::uint8_t>{120U, 100U, 80U,
                                                              255U, 255U, 20U,
                                                              0U, 128U},
              "temperature rgba") &&
       ok;

  auto tint = make_rgba_frame({100U, 100U, 100U, 255U, 10U, 250U, 30U, 128U});
  lmp::filters::TintFilter{-30.0}.process(tint);
  ok = expect(bytes(tint) == std::vector<std::uint8_t>{100U, 70U, 100U, 255U,
                                                       10U, 220U, 30U, 128U},
              "tint rgba") &&
       ok;
  return ok ? 0 : 1;
}
