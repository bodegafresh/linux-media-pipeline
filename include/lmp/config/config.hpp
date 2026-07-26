#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace lmp::config {

using Scalar = std::variant<bool, int, double, std::string>;

struct CaptureConfig {
  std::string type;
  std::string address;
};

struct GpuConfig {
  std::string backend;
};

struct AiConfig {
  std::string engine;
  std::string model_path;
};

struct PipelineConfig {
  std::string threads;
  std::size_t queue_size;
};

struct FilterConfig {
  std::string type;
  bool enabled;
  std::unordered_map<std::string, Scalar> parameters;
};

struct OutputConfig {
  std::string type;
  std::string device;
  std::string pixel_format;
  std::size_t width;
  std::size_t height;
  std::size_t fps;
};

struct AppConfig {
  CaptureConfig capture;
  GpuConfig gpu;
  AiConfig ai;
  PipelineConfig pipeline;
  std::vector<FilterConfig> filters;
  OutputConfig output;
};

} // namespace lmp::config
