#include "lmp/ai/onnx_runtime_engine.hpp"
#include "lmp/capture/gopro_udp_source.hpp"
#include "lmp/config/config_loader.hpp"
#include "lmp/decoder/ffmpeg_decoder.hpp"
#include "lmp/filters/filter_pipeline.hpp"
#include "lmp/filters/filter_registry.hpp"
#include "lmp/frame/frame.hpp"
#include "lmp/output/v4l2_output.hpp"
#include "lmp/version.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

namespace {

void print_help() {
  std::cout
      << "linux-media-pipeline options:\n"
      << "  --help           Show this help.\n"
      << "  --config PATH    Load YAML config. Default: config/default.yaml.\n"
      << "  --open-capture   Bind the configured GoPro UDP listener.\n"
      << "  --check-output   Open the configured V4L2 output device.\n"
      << "  --stream-live    Decode configured capture with FFmpeg and stream "
         "to V4L2.\n"
      << "  --test-pattern   Stream a live RGB test pattern to V4L2 for OBS.\n"
      << "  --stats-every N  Print runtime FPS/latency stats every N "
         "seconds.\n"
      << "  --list-onnx-providers\n"
         "                   List providers exposed by the active ONNX "
         "Runtime build.\n"
      << "  --verify-onnx-gpu\n"
         "                   Run a conservative ONNX provider verification "
         "and write artifacts/onnx-gpu-verification.json.\n"
      << "  --onnx-provider PROVIDER\n"
         "                   Provider for --verify-onnx-gpu: auto, cpu, "
         "migraphx, rocm, or openvino.\n"
      << "  --diagnose-frame PATH\n"
         "                   Process a binary PPM P6 frame and write local "
         "diagnostic artifacts.\n"
      << "  --diagnose-output DIR\n"
         "                   Output directory for --diagnose-frame. Default: "
         "artifacts/frame-diagnostics.\n"
      << "  --segment-diagnostics PATH\n"
         "                   Run ONNX segmentation diagnostics on a PPM P6 "
         "frame.\n"
      << "  --segment-output DIR\n"
         "                   Output directory for --segment-diagnostics. "
         "Default: artifacts/segmentation-diagnostics.\n"
      << "  --segment-model PATH[,PATH...]\n"
         "                   ONNX model path(s) to compare. Defaults to config "
         "AI model.\n"
      << "  --segment-providers PROVIDER[,PROVIDER...]\n"
         "                   Providers to compare. Default: cpu,rocm.\n";
}

void list_onnx_providers() {
  std::cout << "ONNX Runtime version: "
            << lmp::ai::OnnxRuntimeEngine::runtime_version() << "\n\n";
  std::cout << "Available providers:\n";
  for (const auto &provider : lmp::ai::OnnxRuntimeEngine::provider_infos()) {
    if (provider.selectable) {
      std::cout << "  - " << provider.name << '\n';
    }
  }
  std::cout << '\n';
  for (const auto &provider : lmp::ai::OnnxRuntimeEngine::provider_infos()) {
    if (provider.selectable) {
      continue;
    }
    std::cout << provider.name << ":\n"
              << "  compiled_in: " << (provider.compiled_in ? "true" : "false")
              << '\n'
              << "  selectable: " << (provider.selectable ? "true" : "false")
              << '\n'
              << "  reason: " << provider.unavailable_reason << '\n';
  }
}

lmp::frame::Frame make_test_pattern(std::uint32_t width, std::uint32_t height,
                                    std::uint32_t frame_index);

bool bool_parameter(const lmp::config::FilterConfig &filter,
                    std::string_view name, bool default_value);

std::string string_parameter(const lmp::config::FilterConfig &filter,
                             std::string_view name,
                             std::string_view default_value);

std::vector<std::string> split_csv(std::string_view value) {
  auto result = std::vector<std::string>{};
  auto current = std::string{};
  for (const auto character : value) {
    if (character == ',') {
      if (!current.empty()) {
        result.push_back(current);
        current.clear();
      }
      continue;
    }
    if (std::isspace(static_cast<unsigned char>(character)) == 0) {
      current.push_back(character);
    }
  }
  if (!current.empty()) {
    result.push_back(current);
  }
  return result;
}

std::string json_escape(std::string_view value) {
  auto result = std::string{};
  for (const auto character : value) {
    if (character == '"' || character == '\\') {
      result.push_back('\\');
      result.push_back(character);
    } else if (character == '\n') {
      result += "\\n";
    } else {
      result.push_back(character);
    }
  }
  return result;
}

std::string safe_artifact_name(std::string_view value) {
  auto result = std::string{};
  for (const auto character : value) {
    if (std::isalnum(static_cast<unsigned char>(character)) != 0 ||
        character == '-' || character == '_') {
      result.push_back(character);
    } else {
      result.push_back('_');
    }
  }
  return result.empty() ? "artifact" : result;
}

void verify_onnx_provider(const lmp::config::AppConfig &config,
                          std::string_view provider_override) {
  auto model_path = config.ai.model_path;
  auto input_shape = std::string{"1x3x256x256"};
  auto output_shape = std::string{"1x1x256x256"};
  auto provider =
      std::string{provider_override.empty() ? "migraphx" : provider_override};
  auto allow_fallback = true;
  for (const auto &filter : config.filters) {
    if (filter.type != "background_blur") {
      continue;
    }
    model_path = string_parameter(filter, "model_path", model_path);
    input_shape = string_parameter(filter, "input_shape", input_shape);
    output_shape = string_parameter(filter, "output_shape", output_shape);
    allow_fallback = bool_parameter(filter, "allow_provider_fallback", true);
  }

  auto frame = make_test_pattern(1280U, 720U, 0U);
  auto engine = lmp::ai::OnnxRuntimeEngine{
      model_path,     1U,   0.0, input_shape, output_shape, provider,
      allow_fallback, "CPU"};
  const auto model_loaded = engine.available();
  const auto warmup_runs = 5U;
  const auto measured_runs = 20U;
  auto successful_runs = 0U;
  auto total_ms = 0.0;
  auto total_preprocess_ms = 0.0;
  auto total_onnx_run_ms = 0.0;
  auto total_postprocess_ms = 0.0;
  auto p95_source = std::vector<double>{};
  p95_source.reserve(measured_runs);

  for (auto index = 0U; index < warmup_runs && model_loaded; ++index) {
    static_cast<void>(engine.segment_person_blocking(frame));
  }
  for (auto index = 0U; index < measured_runs && model_loaded; ++index) {
    const auto started = std::chrono::steady_clock::now();
    const auto mask = engine.segment_person_blocking(frame);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(elapsed).count();
    const auto finite_output = mask.width() > 0U && mask.height() > 0U;
    if (finite_output) {
      ++successful_runs;
      total_ms += elapsed_ms;
      const auto timing = engine.last_timing();
      total_preprocess_ms += timing.preprocess_ms;
      total_onnx_run_ms += timing.inference_ms;
      total_postprocess_ms += timing.postprocess_ms;
      p95_source.push_back(elapsed_ms);
    }
  }
  std::sort(p95_source.begin(), p95_source.end());
  const auto average_ms = successful_runs == 0U
                              ? 0.0
                              : total_ms / static_cast<double>(successful_runs);
  const auto average_preprocess_ms =
      successful_runs == 0U
          ? 0.0
          : total_preprocess_ms / static_cast<double>(successful_runs);
  const auto average_onnx_run_ms =
      successful_runs == 0U
          ? 0.0
          : total_onnx_run_ms / static_cast<double>(successful_runs);
  const auto average_postprocess_ms =
      successful_runs == 0U
          ? 0.0
          : total_postprocess_ms / static_cast<double>(successful_runs);
  const auto p95_ms =
      p95_source.empty()
          ? 0.0
          : p95_source[std::min(
                p95_source.size() - 1U,
                static_cast<std::size_t>((p95_source.size() * 95U) / 100U))];
  const auto provider_active = std::string{engine.active_provider()};
  const auto inference_successful = successful_runs == measured_runs;
  const auto gpu_verified = (provider_active == "MIGraphXExecutionProvider" ||
                             provider_active == "ROCMExecutionProvider") &&
                            inference_successful && !engine.provider_fallback();
  std::filesystem::create_directories("artifacts");
  std::ofstream report{"artifacts/onnx-gpu-verification.json"};
  report << "{\n"
         << "  \"onnx_runtime_version\": \""
         << lmp::ai::OnnxRuntimeEngine::runtime_version() << "\",\n"
         << "  \"provider_requested\": \"" << provider << "\",\n"
         << "  \"provider_active\": \"" << provider_active << "\",\n"
         << "  \"gpu_execution_verified\": "
         << (gpu_verified ? "true" : "false") << ",\n"
         << "  \"model_loaded\": " << (model_loaded ? "true" : "false") << ",\n"
         << "  \"inference_successful\": "
         << (inference_successful ? "true" : "false") << ",\n"
         << "  \"warmup_runs\": " << warmup_runs << ",\n"
         << "  \"measured_runs\": " << measured_runs << ",\n"
         << "  \"successful_runs\": " << successful_runs << ",\n"
         << "  \"average_inference_ms\": " << average_ms << ",\n"
         << "  \"average_preprocess_ms\": " << average_preprocess_ms << ",\n"
         << "  \"average_onnx_run_ms\": " << average_onnx_run_ms << ",\n"
         << "  \"average_postprocess_ms\": " << average_postprocess_ms << ",\n"
         << "  \"p95_inference_ms\": " << p95_ms << ",\n"
         << "  \"fallback\": "
         << (engine.provider_fallback() ? "true" : "false") << ",\n"
         << "  \"fallback_reason\": \"" << engine.provider_fallback_reason()
         << "\"\n"
         << "}\n";

  std::cout << "onnx_runtime_provider_requested=" << provider << '\n'
            << "onnx_runtime_provider_active=" << provider_active << '\n'
            << "onnx_runtime_provider_fallback="
            << (engine.provider_fallback() ? "true" : "false") << '\n'
            << "onnx_runtime_provider_fallback_reason="
            << engine.provider_fallback_reason() << '\n'
            << "model_loaded=" << (model_loaded ? "true" : "false") << '\n'
            << "inference_successful="
            << (inference_successful ? "true" : "false") << '\n'
            << "gpu_execution_verified=" << (gpu_verified ? "true" : "false")
            << '\n'
            << "average_inference_ms=" << average_ms << '\n'
            << "average_preprocess_ms=" << average_preprocess_ms << '\n'
            << "average_onnx_run_ms=" << average_onnx_run_ms << '\n'
            << "average_postprocess_ms=" << average_postprocess_ms << '\n'
            << "p95_inference_ms=" << p95_ms << '\n'
            << "wrote artifacts/onnx-gpu-verification.json\n";
}

std::optional<double> parse_positive_double(std::string_view value,
                                            std::string_view name) {
  try {
    const auto parsed = std::stod(std::string{value});
    if (parsed <= 0.0) {
      throw std::runtime_error(std::string{name} + " must be > 0");
    }
    return parsed;
  } catch (const std::invalid_argument &) {
    throw std::runtime_error(std::string{name} + " must be a number");
  } catch (const std::out_of_range &) {
    throw std::runtime_error(std::string{name} + " is out of range");
  }
}

std::optional<double> stats_interval_from_env() {
  const auto *configured = std::getenv("LMP_STATS_EVERY");
  if (configured == nullptr || std::string_view{configured}.empty()) {
    return std::nullopt;
  }
  return parse_positive_double(configured, "LMP_STATS_EVERY");
}

class StatsReporter {
public:
  explicit StatsReporter(std::optional<double> interval_seconds)
      : interval_seconds_(interval_seconds),
        window_started_(std::chrono::steady_clock::now()) {}

  void observe(std::chrono::steady_clock::duration processing_time) {
    if (!interval_seconds_.has_value()) {
      return;
    }
    ++frames_;
    total_processing_time_ += processing_time;
    max_processing_time_ = std::max(max_processing_time_, processing_time);

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - window_started_;
    const auto elapsed_seconds = std::chrono::duration<double>(elapsed).count();
    if (elapsed_seconds < *interval_seconds_) {
      return;
    }

    const auto average_ms =
        std::chrono::duration<double, std::milli>(total_processing_time_)
            .count() /
        static_cast<double>(frames_);
    const auto max_ms =
        std::chrono::duration<double, std::milli>(max_processing_time_).count();
    const auto fps = static_cast<double>(frames_) / elapsed_seconds;
    std::cout << "runtime_stats fps=" << fps << " avg_frame_ms=" << average_ms
              << " max_frame_ms=" << max_ms << " frames=" << frames_ << '\n';

    frames_ = 0U;
    total_processing_time_ = std::chrono::steady_clock::duration::zero();
    max_processing_time_ = std::chrono::steady_clock::duration::zero();
    window_started_ = now;
  }

private:
  std::optional<double> interval_seconds_;
  std::chrono::steady_clock::time_point window_started_;
  std::uint64_t frames_ = 0U;
  std::chrono::steady_clock::duration total_processing_time_ =
      std::chrono::steady_clock::duration::zero();
  std::chrono::steady_clock::duration max_processing_time_ =
      std::chrono::steady_clock::duration::zero();
};

class RuntimeMetadataReporter {
public:
  void report(const lmp::frame::Frame::Metadata &metadata) {
    report_once(metadata, "filter_backend", "filter_backend_active");
    report_once(metadata, "background_blur_backend",
                "background_blur_backend_active");
    report_once(metadata, "background_blur_mask",
                "background_blur_mask_active");
    report_once(metadata, "onnx_runtime_available_providers",
                "onnx_runtime_available_providers");
    report_once(metadata, "onnx_runtime_provider_requested",
                "onnx_runtime_provider_requested");
    report_once(metadata, "onnx_runtime_provider_active",
                "onnx_runtime_provider_active");
    report_once(metadata, "onnx_runtime_provider_fallback",
                "onnx_runtime_provider_fallback");
    report_once(metadata, "onnx_runtime_provider_fallback_reason",
                "onnx_runtime_provider_fallback_reason");
    report_once(metadata, "onnx_runtime_model", "onnx_runtime_model");
    report_once(metadata, "openvino_available_devices",
                "openvino_available_devices");
    report_once(metadata, "openvino_device_requested",
                "openvino_device_requested");
    report_once(metadata, "openvino_device_active", "openvino_device_active");
    report_once(metadata, "opencl_platform_active", "opencl_platform_active");
    report_once(metadata, "opencl_device_active", "opencl_device_active");
    report_once(metadata, "segmentation_inference_backend",
                "segmentation_inference_backend");
    report_once(metadata, "segmentation_inference_device",
                "segmentation_inference_device");
    report_once(metadata, "segmentation_mask_inverted",
                "segmentation_mask_inverted");
    report_once(metadata, "segmentation_mask_coverage_raw",
                "segmentation_mask_coverage_raw");
    report_once(metadata, "segmentation_mask_foreground_threshold",
                "segmentation_mask_foreground_threshold");
    report_once(metadata, "segmentation_mask_largest_component",
                "segmentation_mask_largest_component");
    report_once(metadata, "segmentation_mask_coverage_component",
                "segmentation_mask_coverage_component");
    report_once(metadata, "segmentation_mask_quality",
                "segmentation_mask_quality");
    report_once(metadata, "segmentation_mask_adaptive_threshold",
                "segmentation_mask_adaptive_threshold");
    report_once(metadata, "segmentation_mask_coverage_adaptive",
                "segmentation_mask_coverage_adaptive");
    report_once(metadata, "segmentation_mask_recovered",
                "segmentation_mask_recovered");
    report_once(metadata, "segmentation_mask_recovery_rejected",
                "segmentation_mask_recovery_rejected");
    report_once(metadata, "segmentation_mask_rejected",
                "segmentation_mask_rejected");
    report_once(metadata, "segmentation_mask_bad_streak",
                "segmentation_mask_bad_streak");
    report_once(metadata, "segmentation_mask_cooldown_frames",
                "segmentation_mask_cooldown_frames");
    report_once(metadata, "segmentation_mask_hint_coverage",
                "segmentation_mask_hint_coverage");
    report_once(metadata, "segmentation_mask_coverage_refined",
                "segmentation_mask_coverage_refined");
    report_once(metadata, "onnx_preprocess_ms", "onnx_preprocess_ms");
    report_once(metadata, "onnx_inference_ms", "onnx_inference_ms");
    report_once(metadata, "onnx_postprocess_ms", "onnx_postprocess_ms");
    report_once(metadata, "background_processing_backend",
                "background_processing_backend");
    report_once(metadata, "background_processing_device",
                "background_processing_device");
    report_once(metadata, "background_blur_radius", "background_blur_radius");
  }

private:
  void report_once(const lmp::frame::Frame::Metadata &metadata,
                   std::string_view key, std::string_view label) {
    if (reported_.end() !=
        std::find(reported_.begin(), reported_.end(), std::string{key})) {
      return;
    }
    const auto found = metadata.find(std::string{key});
    if (found == metadata.end() || found->second.empty()) {
      return;
    }
    std::cout << label << '=' << found->second << '\n';
    reported_.push_back(std::string{key});
  }

  std::vector<std::string> reported_;
};

std::string
active_filter_list(const std::vector<lmp::config::FilterConfig> &filters) {
  std::string result = "[";
  auto first = true;
  for (const auto &filter : filters) {
    if (!filter.enabled) {
      continue;
    }
    if (!first) {
      result += ",";
    }
    result += filter.type;
    first = false;
  }
  result += "]";
  return result;
}

bool bool_parameter(const lmp::config::FilterConfig &filter,
                    std::string_view name, bool default_value) {
  const auto found = filter.parameters.find(std::string{name});
  if (found == filter.parameters.end()) {
    return default_value;
  }
  if (const auto *value = std::get_if<bool>(&found->second)) {
    return *value;
  }
  return default_value;
}

std::string string_parameter(const lmp::config::FilterConfig &filter,
                             std::string_view name,
                             std::string_view default_value) {
  const auto found = filter.parameters.find(std::string{name});
  if (found == filter.parameters.end()) {
    return std::string{default_value};
  }
  if (const auto *value = std::get_if<std::string>(&found->second)) {
    return *value;
  }
  return std::string{default_value};
}

std::string
pipeline_plan(const std::vector<lmp::config::FilterConfig> &filters) {
  auto cpu_filters = 0U;
  auto gpu_filters = 0U;
  auto fused_filters = 0U;
  for (const auto &filter : filters) {
    if (!filter.enabled) {
      continue;
    }
    const auto backend = string_parameter(filter, "backend", "cpu");
    if (backend == "opencl") {
      ++gpu_filters;
    } else {
      ++cpu_filters;
    }
    if (filter.type == "background_blur" &&
        bool_parameter(filter, "auto_frame", false)) {
      ++fused_filters;
    }
  }

  auto result = std::string{"cpu_filters="} + std::to_string(cpu_filters) +
                " gpu_filters=" + std::to_string(gpu_filters) +
                " fused_filters=" + std::to_string(fused_filters);
  if (fused_filters > 0U && cpu_filters == 0U) {
    result += " mode=fused_gpu";
  } else if (gpu_filters > 0U && cpu_filters > 0U) {
    result += " mode=hybrid";
  } else if (gpu_filters > 0U) {
    result += " mode=gpu";
  } else {
    result += " mode=cpu";
  }
  return result;
}

lmp::frame::Frame make_test_pattern(std::uint32_t width, std::uint32_t height,
                                    std::uint32_t frame_index) {
  std::vector<std::uint8_t> bytes;
  bytes.resize(static_cast<std::size_t>(width) * height * 3U);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const auto offset = ((static_cast<std::size_t>(y) * width) + x) * 3U;
      const auto bar = ((x + frame_index * 8U) / 80U) % 6U;
      bytes[offset] = bar == 0U || bar == 3U ? 255U : 32U;
      bytes[offset + 1U] = bar == 1U || bar == 3U || bar == 4U ? 255U : 32U;
      bytes[offset + 2U] = bar == 2U || bar == 4U || bar == 5U ? 255U : 32U;
      if (((x / 32U) + (y / 32U) + frame_index) % 2U == 0U) {
        bytes[offset] = static_cast<std::uint8_t>(bytes[offset] / 2U);
      }
    }
  }
  return lmp::frame::Frame{
      width,
      height,
      lmp::frame::PixelFormat::Rgb,
      std::move(bytes),
      std::vector<std::size_t>{static_cast<std::size_t>(width) * 3U},
      lmp::frame::Frame::Clock::now()};
}

std::string read_ppm_token(std::istream &input) {
  auto token = std::string{};
  while (input.good()) {
    const auto peeked = input.peek();
    if (peeked == '#') {
      input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    if (std::isspace(peeked) != 0) {
      static_cast<void>(input.get());
      continue;
    }
    break;
  }
  while (input.good() && std::isspace(input.peek()) == 0) {
    token.push_back(static_cast<char>(input.get()));
  }
  if (token.empty()) {
    throw std::runtime_error("invalid PPM: missing token");
  }
  return token;
}

lmp::frame::Frame read_ppm_frame(const std::filesystem::path &path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error("cannot open diagnostic frame: " + path.string());
  }
  if (read_ppm_token(input) != "P6") {
    throw std::runtime_error("diagnostic frame must be binary PPM P6");
  }
  const auto width = static_cast<std::uint32_t>(
      std::stoul(read_ppm_token(input)));
  const auto height = static_cast<std::uint32_t>(
      std::stoul(read_ppm_token(input)));
  const auto max_value = std::stoul(read_ppm_token(input));
  if (max_value != 255U) {
    throw std::runtime_error("diagnostic PPM must use max value 255");
  }
  if (std::isspace(input.peek()) != 0) {
    static_cast<void>(input.get());
  }
  auto bytes = std::vector<std::uint8_t>(
      static_cast<std::size_t>(width) * height * 3U);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error("diagnostic PPM data is incomplete");
  }
  return lmp::frame::Frame{
      width,
      height,
      lmp::frame::PixelFormat::Rgb,
      std::move(bytes),
      std::vector<std::size_t>{static_cast<std::size_t>(width) * 3U},
      lmp::frame::Frame::Clock::now()};
}

void write_ppm_frame(const std::filesystem::path &path,
                     const lmp::frame::Frame &frame) {
  if (frame.format() != lmp::frame::PixelFormat::Rgb) {
    throw std::runtime_error("diagnostic PPM writer requires RGB frames");
  }
  std::ofstream output{path, std::ios::binary};
  if (!output) {
    throw std::runtime_error("cannot write diagnostic frame: " + path.string());
  }
  output << "P6\n" << frame.width() << ' ' << frame.height() << "\n255\n";
  const auto bytes = frame.data();
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

void write_pgm_mask(const std::filesystem::path &path,
                    const lmp::ai::SegmentationMask &mask) {
  std::ofstream output{path, std::ios::binary};
  if (!output) {
    throw std::runtime_error("cannot write diagnostic mask: " + path.string());
  }
  output << "P5\n" << mask.width() << ' ' << mask.height() << "\n255\n";
  const auto values = mask.values();
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size()));
}

void write_mask_overlay(const std::filesystem::path &path,
                        const lmp::frame::Frame &frame,
                        const lmp::ai::SegmentationMask &mask,
                        std::uint8_t threshold) {
  if (frame.format() != lmp::frame::PixelFormat::Rgb) {
    throw std::runtime_error("mask overlay writer requires RGB frames");
  }
  auto output_frame = frame;
  const auto original_bytes = output_frame.data();
  auto bytes =
      std::vector<std::uint8_t>{original_bytes.begin(), original_bytes.end()};
  for (std::uint32_t y = 0; y < frame.height(); ++y) {
    const auto mask_y = std::min(
        (static_cast<std::uint64_t>(y) * mask.height()) / frame.height(),
        static_cast<std::uint64_t>(mask.height() - 1U));
    for (std::uint32_t x = 0; x < frame.width(); ++x) {
      const auto mask_x = std::min(
          (static_cast<std::uint64_t>(x) * mask.width()) / frame.width(),
          static_cast<std::uint64_t>(mask.width() - 1U));
      const auto mask_value = mask.at(static_cast<std::uint32_t>(mask_x),
                                      static_cast<std::uint32_t>(mask_y));
      const auto offset = ((static_cast<std::size_t>(y) * frame.width()) + x) *
                          static_cast<std::size_t>(3U);
      if (mask_value >= threshold) {
        bytes[offset + 1U] = static_cast<std::uint8_t>(
            std::min(255U, static_cast<unsigned>(bytes[offset + 1U]) + 90U));
      } else {
        bytes[offset] = static_cast<std::uint8_t>(
            static_cast<unsigned>(bytes[offset]) / 2U);
        bytes[offset + 1U] = static_cast<std::uint8_t>(
            static_cast<unsigned>(bytes[offset + 1U]) / 2U);
        bytes[offset + 2U] = static_cast<std::uint8_t>(
            std::min(255U, static_cast<unsigned>(bytes[offset + 2U]) + 50U));
      }
    }
  }
  output_frame = lmp::frame::Frame{
      frame.width(),
      frame.height(),
      lmp::frame::PixelFormat::Rgb,
      std::move(bytes),
      std::vector<std::size_t>{static_cast<std::size_t>(frame.width()) * 3U},
      frame.timestamp()};
  write_ppm_frame(path, output_frame);
}

double mask_mean_absolute_difference(const lmp::ai::SegmentationMask &left,
                                     const lmp::ai::SegmentationMask &right) {
  if (left.width() != right.width() || left.height() != right.height()) {
    return -1.0;
  }
  auto sum = std::uint64_t{0U};
  for (std::size_t index = 0; index < left.values().size(); ++index) {
    const auto l = static_cast<int>(left.values()[index]);
    const auto r = static_cast<int>(right.values()[index]);
    sum += static_cast<std::uint64_t>(std::abs(l - r));
  }
  return static_cast<double>(sum) /
         (static_cast<double>(left.values().size()) * 255.0);
}

void segment_diagnostics(const lmp::config::AppConfig &config,
                         const std::filesystem::path &input_path,
                         const std::filesystem::path &output_dir,
                         const std::vector<std::string> &model_paths,
                         const std::vector<std::string> &providers) {
  constexpr auto kMaskThreshold = std::uint8_t{180U};
  std::filesystem::create_directories(output_dir);
  const auto frame = read_ppm_frame(input_path);
  write_ppm_frame(output_dir / "input.ppm", frame);

  auto report_entries = std::vector<std::string>{};
  auto baseline_masks = std::vector<std::pair<std::string, lmp::ai::SegmentationMask>>{};

  for (const auto &model_path : model_paths) {
    const auto model_name =
        safe_artifact_name(std::filesystem::path{model_path}.stem().string());
    auto cpu_mask = std::optional<lmp::ai::SegmentationMask>{};
    for (const auto &provider : providers) {
      auto engine = lmp::ai::OnnxRuntimeEngine{
          model_path, 1U, 0.0, "1x3x256x256", "1x1x256x256", provider, true,
          "CPU"};
      const auto started = std::chrono::steady_clock::now();
      const auto mask = engine.segment_person_blocking(frame);
      const auto elapsed = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started)
                               .count();
      const auto timing = engine.last_timing();
      const auto coverage = lmp::ai::mask_coverage(mask, kMaskThreshold);
      const auto artifact_prefix =
          model_name + "_" + safe_artifact_name(provider);
      write_pgm_mask(output_dir / (artifact_prefix + "_mask.pgm"), mask);
      write_mask_overlay(output_dir / (artifact_prefix + "_overlay.ppm"),
                         frame, mask, kMaskThreshold);
      auto cpu_diff = -1.0;
      if (provider == "cpu") {
        cpu_mask = mask;
        baseline_masks.emplace_back(model_name, mask);
      } else if (cpu_mask.has_value()) {
        cpu_diff = mask_mean_absolute_difference(*cpu_mask, mask);
      } else {
        const auto found = std::find_if(
            baseline_masks.begin(), baseline_masks.end(),
            [&](const auto &entry) { return entry.first == model_name; });
        if (found != baseline_masks.end()) {
          cpu_diff = mask_mean_absolute_difference(found->second, mask);
        }
      }

      report_entries.push_back(
          "    {\n"
          "      \"model\": \"" +
          json_escape(model_path) +
          "\",\n"
          "      \"provider_requested\": \"" +
          json_escape(provider) +
          "\",\n"
          "      \"provider_active\": \"" +
          json_escape(engine.active_provider()) +
          "\",\n"
          "      \"provider_fallback\": " +
          std::string{engine.provider_fallback() ? "true" : "false"} +
          ",\n"
          "      \"fallback_reason\": \"" +
          json_escape(engine.provider_fallback_reason()) +
          "\",\n"
          "      \"available_providers\": \"" +
          json_escape(engine.available_providers()) +
          "\",\n"
          "      \"model_summary\": \"" + json_escape(engine.model_summary()) +
          "\",\n"
          "      \"coverage\": " +
          std::to_string(coverage) +
          ",\n"
          "      \"cpu_mask_mae\": " +
          std::to_string(cpu_diff) +
          ",\n"
          "      \"elapsed_ms\": " +
          std::to_string(elapsed) +
          ",\n"
          "      \"preprocess_ms\": " +
          std::to_string(timing.preprocess_ms) +
          ",\n"
          "      \"inference_ms\": " +
          std::to_string(timing.inference_ms) +
          ",\n"
          "      \"postprocess_ms\": " +
          std::to_string(timing.postprocess_ms) +
          ",\n"
          "      \"mask_path\": \"" +
          json_escape((output_dir / (artifact_prefix + "_mask.pgm")).string()) +
          "\",\n"
          "      \"overlay_path\": \"" +
          json_escape((output_dir / (artifact_prefix + "_overlay.ppm")).string()) +
          "\"\n"
          "    }");
    }
  }

  std::ofstream report{output_dir / "report.json"};
  report << "{\n"
         << "  \"input\": \"" << json_escape(input_path.string()) << "\",\n"
         << "  \"width\": " << frame.width() << ",\n"
         << "  \"height\": " << frame.height() << ",\n"
         << "  \"threshold\": " << static_cast<unsigned>(kMaskThreshold)
         << ",\n"
         << "  \"results\": [\n";
  for (std::size_t index = 0; index < report_entries.size(); ++index) {
    if (index > 0U) {
      report << ",\n";
    }
    report << report_entries[index];
  }
  report << "\n  ]\n}\n";

  std::cout << "wrote " << (output_dir / "input.ppm").string() << '\n'
            << "wrote " << (output_dir / "report.json").string() << '\n';
  static_cast<void>(config);
}

void diagnose_frame(const lmp::config::AppConfig &config,
                    const lmp::filters::FilterPipeline &pipeline,
                    const std::filesystem::path &input_path,
                    const std::filesystem::path &output_dir) {
  std::filesystem::create_directories(output_dir);
  auto input_frame = read_ppm_frame(input_path);
  auto output_frame = input_frame;
  const auto started = std::chrono::steady_clock::now();
  pipeline.process(output_frame);
  const auto elapsed = std::chrono::steady_clock::now() - started;

  write_ppm_frame(output_dir / "input.ppm", input_frame);
  write_ppm_frame(output_dir / "output.ppm", output_frame);

  std::ofstream metadata{output_dir / "metadata.txt"};
  metadata << "config=" << config.capture.type << '\n'
           << "input=" << input_path.string() << '\n'
           << "width=" << output_frame.width() << '\n'
           << "height=" << output_frame.height() << '\n'
           << "processing_ms="
           << std::chrono::duration<double, std::milli>(elapsed).count()
           << '\n';
  for (const auto &[key, value] : output_frame.metadata()) {
    metadata << key << '=' << value << '\n';
  }
  std::cout << "wrote " << (output_dir / "input.ppm").string() << '\n'
            << "wrote " << (output_dir / "output.ppm").string() << '\n'
            << "wrote " << (output_dir / "metadata.txt").string() << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    auto config_path = std::string{"config/default.yaml"};
    auto open_capture = false;
    auto check_output = false;
    auto stream_live = false;
    auto test_pattern = false;
    auto list_providers = false;
    auto verify_onnx_gpu = false;
    auto onnx_provider = std::string{"migraphx"};
    auto diagnose_frame_path = std::optional<std::filesystem::path>{};
    auto diagnose_output_dir =
        std::filesystem::path{"artifacts/frame-diagnostics"};
    auto segment_diagnostics_path = std::optional<std::filesystem::path>{};
    auto segment_output_dir =
        std::filesystem::path{"artifacts/segmentation-diagnostics"};
    auto segment_models = std::vector<std::string>{};
    auto segment_providers = std::vector<std::string>{"cpu", "rocm"};
    auto stats_every = stats_interval_from_env();
    for (int index = 1; index < argc; ++index) {
      const auto option = std::string_view{argv[index]};
      if (option == "--help") {
        print_help();
        return 0;
      }
      if (option == "--config") {
        ++index;
        if (index >= argc) {
          throw std::runtime_error("--config requires a path");
        }
        config_path = argv[index];
      } else if (option == "--stats-every") {
        ++index;
        if (index >= argc) {
          throw std::runtime_error("--stats-every requires seconds");
        }
        stats_every = parse_positive_double(argv[index], "--stats-every");
      } else if (option == "--open-capture") {
        open_capture = true;
      } else if (option == "--check-output") {
        check_output = true;
      } else if (option == "--stream-live") {
        stream_live = true;
      } else if (option == "--test-pattern") {
        test_pattern = true;
      } else if (option == "--list-onnx-providers") {
        list_providers = true;
      } else if (option == "--verify-onnx-gpu") {
        verify_onnx_gpu = true;
      } else if (option == "--onnx-provider") {
        ++index;
        if (index >= argc) {
          throw std::runtime_error("--onnx-provider requires a provider name");
        }
        onnx_provider = argv[index];
      } else if (option == "--diagnose-frame") {
        ++index;
        if (index >= argc) {
          throw std::runtime_error("--diagnose-frame requires a PPM path");
        }
        diagnose_frame_path = std::filesystem::path{argv[index]};
      } else if (option == "--diagnose-output") {
        ++index;
        if (index >= argc) {
          throw std::runtime_error("--diagnose-output requires a directory");
        }
        diagnose_output_dir = std::filesystem::path{argv[index]};
      } else if (option == "--segment-diagnostics") {
        ++index;
        if (index >= argc) {
          throw std::runtime_error("--segment-diagnostics requires a PPM path");
        }
        segment_diagnostics_path = std::filesystem::path{argv[index]};
      } else if (option == "--segment-output") {
        ++index;
        if (index >= argc) {
          throw std::runtime_error("--segment-output requires a directory");
        }
        segment_output_dir = std::filesystem::path{argv[index]};
      } else if (option == "--segment-model") {
        ++index;
        if (index >= argc) {
          throw std::runtime_error("--segment-model requires a path list");
        }
        segment_models = split_csv(argv[index]);
      } else if (option == "--segment-providers") {
        ++index;
        if (index >= argc) {
          throw std::runtime_error("--segment-providers requires a provider list");
        }
        segment_providers = split_csv(argv[index]);
      } else {
        throw std::runtime_error("unknown option: " + std::string{option});
      }
    }
    if (list_providers) {
      list_onnx_providers();
      return 0;
    }

    const lmp::config::ConfigLoader loader;
    const auto config = loader.load_file(config_path);
    const auto registry = lmp::filters::create_default_registry();
    if (verify_onnx_gpu) {
      verify_onnx_provider(config, onnx_provider);
      return 0;
    }
    if (segment_diagnostics_path.has_value()) {
      if (segment_models.empty()) {
        segment_models.push_back(config.ai.model_path);
      }
      if (segment_providers.empty()) {
        segment_providers = {"cpu", "rocm"};
      }
      std::cout << "linux-media-pipeline " << lmp::version_string()
                << " segment_diagnostics=true config=" << config_path
                << " input=" << segment_diagnostics_path->string()
                << " output_dir=" << segment_output_dir.string() << '\n';
      segment_diagnostics(config, *segment_diagnostics_path,
                          segment_output_dir, segment_models,
                          segment_providers);
      return 0;
    }
    const auto pipeline =
        lmp::filters::FilterPipeline::from_config(config.filters, registry);
    const auto filters_active = active_filter_list(config.filters);
    const auto plan = pipeline_plan(config.filters);
    if (diagnose_frame_path.has_value()) {
      std::cout << "linux-media-pipeline " << lmp::version_string()
                << " diagnose_frame=true config=" << config_path
                << " input=" << diagnose_frame_path->string()
                << " output_dir=" << diagnose_output_dir.string()
                << " filter_backend=requested:" << config.gpu.backend
                << " filters=" << pipeline.size()
                << " filters_active=" << filters_active << " pipeline_plan=\""
                << plan << "\"\n";
      diagnose_frame(config, pipeline, *diagnose_frame_path,
                     diagnose_output_dir);
      return 0;
    }

    lmp::capture::GoProUdpSource capture{config.capture.address};
    if (config.capture.type != capture.type()) {
      throw std::runtime_error("unsupported capture type: " +
                               config.capture.type);
    }
    if (open_capture) {
      capture.open();
    }

    lmp::output::V4l2Output output{config.output.device};
    if (config.output.type != output.type()) {
      throw std::runtime_error("unsupported output type: " +
                               config.output.type);
    }
    if (check_output || test_pattern || stream_live) {
      output.open();
    }
    if (stream_live) {
      if (config.output.pixel_format != "RGB24") {
        throw std::runtime_error("live output requires RGB24 pixel_format");
      }
      const auto width = static_cast<std::uint32_t>(config.output.width);
      const auto height = static_cast<std::uint32_t>(config.output.height);
      const auto fps = static_cast<std::uint32_t>(config.output.fps);
      output.configure_rgb24(width, height, fps);
      lmp::decoder::FfmpegDecoder decoder{config.capture.address, width,
                                          height};
      std::cout << "linux-media-pipeline " << lmp::version_string()
                << " streaming live=true capture=" << capture.type()
                << " input=" << config.capture.address
                << " output=" << output.type() << " device=" << output.device()
                << " format=" << config.output.pixel_format
                << " width=" << width << " height=" << height << " fps=" << fps
                << " filter_backend=requested:" << config.gpu.backend
                << " filters=" << pipeline.size()
                << " filters_active=" << filters_active << " pipeline_plan=\""
                << plan << "\"\n";
      RuntimeMetadataReporter runtime_metadata;
      StatsReporter stats{stats_every};
      while (true) {
        const auto frame_started = std::chrono::steady_clock::now();
        auto frame = decoder.read_frame();
        pipeline.process(frame);
        runtime_metadata.report(frame.metadata());
        output.write(frame);
        stats.observe(std::chrono::steady_clock::now() - frame_started);
      }
    }
    if (test_pattern) {
      if (config.output.pixel_format != "RGB24") {
        throw std::runtime_error(
            "test pattern output requires RGB24 pixel_format");
      }
      const auto width = static_cast<std::uint32_t>(config.output.width);
      const auto height = static_cast<std::uint32_t>(config.output.height);
      const auto fps = static_cast<std::uint32_t>(config.output.fps);
      output.configure_rgb24(width, height, fps);
      std::cout << "linux-media-pipeline " << lmp::version_string()
                << " streaming test_pattern=true output=" << output.type()
                << " device=" << output.device()
                << " format=" << config.output.pixel_format
                << " width=" << width << " height=" << height << " fps=" << fps
                << " filter_backend=requested:" << config.gpu.backend
                << " filters=" << pipeline.size()
                << " filters_active=" << filters_active << " pipeline_plan=\""
                << plan << "\"\n";
      RuntimeMetadataReporter runtime_metadata;
      StatsReporter stats{stats_every};
      for (std::uint32_t frame_index = 0;; ++frame_index) {
        const auto frame_started = std::chrono::steady_clock::now();
        auto frame = make_test_pattern(width, height, frame_index);
        pipeline.process(frame);
        runtime_metadata.report(frame.metadata());
        output.write(frame);
        stats.observe(std::chrono::steady_clock::now() - frame_started);
        std::this_thread::sleep_for(std::chrono::milliseconds{1000 / fps});
      }
    }

    std::cout << "linux-media-pipeline " << lmp::version_string()
              << " capture=" << capture.type()
              << " udp=" << capture.endpoint().host << ':'
              << capture.endpoint().port
              << " capture_open=" << (capture.is_open() ? "true" : "false")
              << " output=" << output.type() << " device=" << output.device()
              << " output_open=" << (output.is_open() ? "true" : "false")
              << " filter_backend=requested:" << config.gpu.backend
              << " filters=" << pipeline.size()
              << " filters_active=" << filters_active << " pipeline_plan=\""
              << plan << "\"\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "linux-media-pipeline: " << error.what() << '\n';
    return 1;
  }
}
