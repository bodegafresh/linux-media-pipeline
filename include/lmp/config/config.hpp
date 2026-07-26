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
};

struct AppConfig {
  CaptureConfig capture;
  GpuConfig gpu;
  PipelineConfig pipeline;
  std::vector<FilterConfig> filters;
  OutputConfig output;
};

} // namespace lmp::config
