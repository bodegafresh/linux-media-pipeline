#include "lmp/ai/onnx_runtime_engine.hpp"
#include "lmp/ai/segmentation_mask.hpp"
#include "lmp/capture/gopro_udp_source.hpp"
#include "lmp/config/config_loader.hpp"
#include "lmp/filters/auto_frame_filter.hpp"
#include "lmp/filters/background_blur_filter.hpp"
#include "lmp/filters/box_blur_filter.hpp"
#include "lmp/filters/brightness_filter.hpp"
#include "lmp/filters/color_adjust_filter.hpp"
#include "lmp/filters/contrast_filter.hpp"
#include "lmp/filters/exposure_filter.hpp"
#include "lmp/filters/filter_pipeline.hpp"
#include "lmp/filters/filter_registry.hpp"
#include "lmp/filters/fps_overlay_filter.hpp"
#include "lmp/filters/gamma_filter.hpp"
#include "lmp/filters/gaussian_blur_filter.hpp"
#include "lmp/filters/grayscale_filter.hpp"
#include "lmp/filters/histogram_filter.hpp"
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
#include "lmp/frame/frame.hpp"
#include "lmp/gpu/buffer_pool.hpp"
#include "lmp/gpu/opencl_backend.hpp"
#include "lmp/gpu/vulkan_backend.hpp"
#include "lmp/output/v4l2_output.hpp"
#include "lmp/version.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
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

lmp::frame::Frame make_rgba_frame(std::uint32_t width, std::uint32_t height,
                                  std::vector<std::uint8_t> data) {
  return lmp::frame::Frame{
      width,
      height,
      lmp::frame::PixelFormat::Rgba,
      std::move(data),
      std::vector<std::size_t>{static_cast<std::size_t>(width) * 4U},
      lmp::frame::Frame::Clock::time_point{std::chrono::milliseconds{1234}}};
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
  const auto endpoint =
      lmp::capture::parse_udp_endpoint(config.capture.address);
  ok = expect(endpoint.host == "0.0.0.0", "gopro udp host") && ok;
  ok = expect(endpoint.port == 8554U, "gopro udp port") && ok;
  ok = expect(config.gpu.backend == "opencl", "gpu backend") && ok;
  ok = expect(config.ai.engine == "onnxruntime", "ai engine") && ok;
  ok = expect(config.ai.model_path == "assets/models/person-segmentation.onnx",
              "ai model path") &&
       ok;
  ok = expect(config.pipeline.threads == "auto", "pipeline threads") && ok;
  ok = expect(config.pipeline.queue_size == 4U, "pipeline queue size") && ok;
  ok = expect(config.filters.size() == 21U, "filter count") && ok;
  ok = expect(config.filters.front().type == "identity",
              "identity filter config") &&
       ok;
  ok = expect(config.filters.front().enabled, "identity filter enabled") && ok;
  ok = expect(config.output.type == "v4l2", "output type") && ok;
  ok = expect(config.output.device == "/dev/video20", "output device") && ok;
  ok = expect(config.output.pixel_format == "RGB24", "output pixel format") &&
       ok;
  ok = expect(config.output.width == 1280U, "output width") && ok;
  ok = expect(config.output.height == 720U, "output height") && ok;
  ok = expect(config.output.fps == 30U, "output fps") && ok;

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
  ok = expect(registry.contains("text_overlay"), "text overlay registered") &&
       ok;
  ok = expect(registry.contains("text"), "text alias registered") && ok;
  ok = expect(registry.contains("fps_overlay"), "fps overlay registered") && ok;
  ok = expect(registry.contains("fps"), "fps alias registered") && ok;
  ok = expect(registry.contains("timestamp_overlay"),
              "timestamp overlay registered") &&
       ok;
  ok = expect(registry.contains("timestamp"), "timestamp alias registered") &&
       ok;
  ok = expect(registry.contains("histogram"), "histogram registered") && ok;
  ok = expect(registry.contains("background_blur"),
              "background blur registered") &&
       ok;
  ok = expect(registry.contains("color_adjust"), "color adjust registered") &&
       ok;
  ok = expect(registry.contains("auto_frame"), "auto frame registered") && ok;
  ok = expect(registry.size() == 27U, "default registry size") && ok;
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

  auto color_adjust =
      make_rgba_frame({100U, 150U, 200U, 255U, 10U, 20U, 30U, 128U});
  lmp::filters::ColorAdjustFilter{2.0, 1.04, 1.03}.process(color_adjust);
  ok = expect(bytes(color_adjust) == std::vector<std::uint8_t>{100U, 153U, 207U,
                                                               255U, 7U, 18U,
                                                               28U, 128U},
              "color adjust rgba") &&
       ok;

  auto text_overlay =
      make_rgba_frame(12U, 6U, std::vector<std::uint8_t>(12U * 6U * 4U, 0U));
  lmp::filters::TextOverlayFilter{"A", 0U, 0U}.process(text_overlay);
  ok = expect(text_overlay.data()[4U] == 255U &&
                  text_overlay.data()[5U] == 255U &&
                  text_overlay.data()[6U] == 255U,
              "text overlay draws glyph") &&
       ok;

  auto fps_overlay =
      make_rgba_frame(24U, 6U, std::vector<std::uint8_t>(24U * 6U * 4U, 0U));
  lmp::filters::FpsOverlayFilter{60.0, 0U, 0U}.process(fps_overlay);
  ok = expect(fps_overlay.data()[1U] == 255U, "fps overlay draws green") && ok;

  auto timestamp_overlay =
      make_rgba_frame(24U, 6U, std::vector<std::uint8_t>(24U * 6U * 4U, 0U));
  lmp::filters::TimestampOverlayFilter{0U, 0U}.process(timestamp_overlay);
  ok = expect(timestamp_overlay.data()[0U] == 255U &&
                  timestamp_overlay.data()[1U] == 255U,
              "timestamp overlay draws yellow") &&
       ok;

  auto histogram = make_rgb_frame(
      4U, 1U, {0U, 0U, 0U, 16U, 16U, 16U, 128U, 128U, 128U, 255U, 255U, 255U});
  lmp::filters::HistogramFilter{false, 0U, 0U, 0U, 0U}.process(histogram);
  ok = expect(histogram.metadata().at("histogram.luma16") ==
                  "1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1",
              "histogram metadata") &&
       ok;

  const lmp::ai::SegmentationMask explicit_mask{2U, 1U, {0U, 255U}};
  const lmp::ai::SegmentationMask previous_mask{2U, 1U, {100U, 100U}};
  const auto blended_mask = explicit_mask.blend_with(previous_mask, 0.5);
  const auto thresholded_mask = lmp::ai::threshold_mask(blended_mask, 128U);
  const lmp::ai::SegmentationMask edge_mask{3U, 1U, {0U, 255U, 0U}};
  const auto expanded_mask = lmp::ai::refine_mask(edge_mask, 128U, 1U, 0U);
  const auto feathered_mask = lmp::ai::refine_mask(edge_mask, 128U, 0U, 1U);
  ok = expect(explicit_mask.width() == 2U, "segmentation mask width") && ok;
  ok = expect(explicit_mask.height() == 1U, "segmentation mask height") && ok;
  ok =
      expect(explicit_mask.at(1U, 0U) == 255U, "segmentation mask value") && ok;
  ok = expect(blended_mask.at(0U, 0U) == 50U && blended_mask.at(1U, 0U) == 178U,
              "segmentation mask blend") &&
       ok;
  ok = expect(thresholded_mask.at(0U, 0U) == 0U &&
                  thresholded_mask.at(1U, 0U) == 255U,
              "segmentation mask threshold") &&
       ok;
  ok = expect(expanded_mask.at(0U, 0U) == 255U &&
                  expanded_mask.at(2U, 0U) == 255U,
              "segmentation mask expansion") &&
       ok;
  ok = expect(feathered_mask.at(0U, 0U) == 128U &&
                  feathered_mask.at(1U, 0U) == 85U &&
                  feathered_mask.at(2U, 0U) == 128U,
              "segmentation mask feather") &&
       ok;

  lmp::ai::OnnxRuntimeEngine onnx{"model.onnx"};
  auto segmentation_frame =
      make_rgb_frame(2U, 1U, {10U, 10U, 10U, 240U, 240U, 240U});
  const auto inferred_mask = onnx.segment_person(segmentation_frame);
  ok = expect(onnx.name() == "onnxruntime", "onnx engine name") && ok;
  ok =
      expect(!onnx.available(), "onnx engine availability without model") && ok;
  ok = expect(!onnx.available_providers().empty(),
              "onnx provider list is reportable") &&
       ok;
  ok = expect(!onnx.requested_provider().empty(),
              "onnx requested provider is reportable") &&
       ok;
  ok = expect(!onnx.active_provider().empty(),
              "onnx active provider is reportable") &&
       ok;
  ok = expect(!onnx.model_summary().empty(),
              "onnx model summary is reportable") &&
       ok;
  ok =
      expect(inferred_mask.at(0U, 0U) == 0U && inferred_mask.at(1U, 0U) == 255U,
             "onnx fallback segmentation") &&
      ok;

  auto background_blur =
      make_rgb_frame(3U, 1U, {0U, 0U, 0U, 255U, 255U, 255U, 90U, 90U, 90U});
  lmp::filters::BackgroundBlurFilter{1U, 128U}.process(background_blur);
  ok = expect(bytes(background_blur) ==
                  std::vector<std::uint8_t>{85U, 85U, 85U, 255U, 255U, 255U,
                                            145U, 145U, 145U},
              "background blur preserves foreground") &&
       ok;
  auto invalid_background_blur = false;
  try {
    lmp::filters::BackgroundBlurFilter{1U, 255U}.process(background_blur);
  } catch (const std::invalid_argument &) {
    invalid_background_blur = true;
  }
  ok = expect(invalid_background_blur,
              "background blur rejects opaque threshold") &&
       ok;

  auto auto_frame = make_rgb_frame(
      6U, 4U,
      {0U,   0U,   0U,   10U,  0U,   0U,   20U,  0U,   0U,   30U,  0U,   0U,
       40U,  0U,   0U,   50U,  0U,   0U,   0U,   10U,  0U,   10U,  10U,  0U,
       20U,  10U,  0U,   30U,  10U,  0U,   240U, 240U, 240U, 250U, 250U, 250U,
       0U,   20U,  0U,   10U,  20U,  0U,   20U,  20U,  0U,   30U,  20U,  0U,
       240U, 240U, 240U, 250U, 250U, 250U, 0U,   30U,  0U,   10U,  30U,  0U,
       20U,  30U,  0U,   30U,  30U,  0U,   40U,  30U,  0U,   50U,  30U,  0U});
  lmp::filters::AutoFrameFilter{0.8, 2.0, 128U}.process(auto_frame);
  ok = expect(auto_frame.metadata().at("auto_frame") == "2,0,4,3",
              "auto frame metadata") &&
       ok;

  auto owned_buffer = lmp::gpu::GpuBuffer::allocate_host(8U);
  owned_buffer.host_span()[0] = 42U;
  ok = expect(owned_buffer.size() == 8U, "owned gpu buffer size") && ok;
  ok = expect(owned_buffer.owns_memory(), "owned gpu buffer ownership") && ok;
  ok = expect(!owned_buffer.zero_copy_capable(),
              "owned gpu buffer not zero copy") &&
       ok;
  ok =
      expect(owned_buffer.host_span()[0] == 42U, "owned gpu buffer writable") &&
      ok;

  std::vector<std::uint8_t> external_memory{1U, 2U, 3U, 4U};
  auto shared_buffer = lmp::gpu::GpuBuffer::wrap_host(external_memory);
  shared_buffer.host_span()[1] = 9U;
  ok = expect(shared_buffer.size() == 4U, "shared gpu buffer size") && ok;
  ok =
      expect(!shared_buffer.owns_memory(), "shared gpu buffer ownership") && ok;
  ok = expect(shared_buffer.zero_copy_capable(),
              "shared gpu buffer zero copy") &&
       ok;
  ok = expect(external_memory[1] == 9U, "shared gpu buffer writes through") &&
       ok;

  lmp::gpu::BufferPool pool{16U, 2U};
  ok = expect(pool.available() == 2U, "buffer pool initial availability") && ok;
  auto first = pool.acquire();
  ok = expect(pool.available() == 1U, "buffer pool acquire") && ok;
  pool.release(std::move(first));
  ok = expect(pool.available() == 2U, "buffer pool release") && ok;

  const lmp::gpu::OpenClBackend opencl;
  auto opencl_buffer = opencl.create_buffer(4U);
  auto imported = opencl.import_host_buffer(external_memory);
  ok = expect(opencl.name() == "opencl", "opencl backend name") && ok;
  ok = expect(opencl.supports_zero_copy_host_memory(),
              "opencl backend zero copy capability") &&
       ok;
  ok = expect(opencl_buffer->owns_memory(), "opencl owned buffer") && ok;
  ok = expect(imported->zero_copy_capable(),
              "opencl imported zero copy buffer") &&
       ok;

  const lmp::gpu::VulkanBackend vulkan;
  auto vulkan_buffer = vulkan.create_buffer(4U);
  auto vulkan_imported = vulkan.import_host_buffer(external_memory);
  ok = expect(vulkan.name() == "vulkan", "vulkan backend name") && ok;
  ok = expect(vulkan.supports_zero_copy_host_memory(),
              "vulkan backend zero copy capability") &&
       ok;
  ok = expect(vulkan_buffer->owns_memory(), "vulkan owned buffer") && ok;
  ok = expect(vulkan_imported->zero_copy_capable(),
              "vulkan imported zero copy buffer") &&
       ok;

  lmp::capture::GoProUdpSource gopro{"udp://127.0.0.1:49152"};
  ok = expect(gopro.type() == "gopro_udp", "gopro source type") && ok;
  ok = expect(!gopro.is_open(), "gopro source starts closed") && ok;
  try {
    gopro.open();
    ok = expect(gopro.is_open(), "gopro source opens udp socket") && ok;
    gopro.close();
    ok = expect(!gopro.is_open(), "gopro source closes udp socket") && ok;
  } catch (const std::runtime_error &error) {
    std::cerr << "SKIPPED: gopro udp bind unavailable in this environment: "
              << error.what() << '\n';
  }

  lmp::output::V4l2Output output{"/dev/null"};
  ok = expect(output.type() == "v4l2", "v4l2 output type") && ok;
  ok = expect(output.device() == "/dev/null", "v4l2 output device") && ok;
  output.open();
  ok = expect(output.is_open(), "v4l2 output opens") && ok;
  const auto black_frame =
      make_rgb_frame(1U, 1U, std::vector<std::uint8_t>{0U, 0U, 0U});
  output.write(black_frame);
  output.close();
  ok = expect(!output.is_open(), "v4l2 output closes") && ok;
  return ok ? 0 : 1;
}
