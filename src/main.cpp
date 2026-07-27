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
#include <cstdint>
#include <cstdlib>
#include <iostream>
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
         "seconds.\n";
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

} // namespace

int main(int argc, char **argv) {
  try {
    auto config_path = std::string{"config/default.yaml"};
    auto open_capture = false;
    auto check_output = false;
    auto stream_live = false;
    auto test_pattern = false;
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
      } else {
        throw std::runtime_error("unknown option: " + std::string{option});
      }
    }

    const lmp::config::ConfigLoader loader;
    const auto config = loader.load_file(config_path);
    const auto registry = lmp::filters::create_default_registry();
    const auto pipeline =
        lmp::filters::FilterPipeline::from_config(config.filters, registry);
    const auto filters_active = active_filter_list(config.filters);
    const auto plan = pipeline_plan(config.filters);

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
