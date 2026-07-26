#include "lmp/config/config_loader.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <variant>

namespace lmp::config {
namespace {

enum class Section {
  None,
  Capture,
  Gpu,
  Ai,
  Pipeline,
  Filters,
  Output,
};

std::string trim(std::string_view value) {
  auto begin = value.begin();
  auto end = value.end();
  while (begin != end &&
         std::isspace(static_cast<unsigned char>(*begin)) != 0) {
    ++begin;
  }
  while (begin != end &&
         std::isspace(static_cast<unsigned char>(*(end - 1))) != 0) {
    --end;
  }
  return std::string(begin, end);
}

bool starts_with(std::string_view text, std::string_view prefix) noexcept {
  return text.size() >= prefix.size() &&
         text.substr(0, prefix.size()) == prefix;
}

std::pair<std::string, std::string> split_key_value(std::string_view line) {
  const auto separator = line.find(':');
  if (separator == std::string_view::npos) {
    throw std::invalid_argument("expected key: value line");
  }
  return {trim(line.substr(0, separator)), trim(line.substr(separator + 1))};
}

Scalar parse_scalar(const std::string &value) {
  if (value == "true") {
    return true;
  }
  if (value == "false") {
    return false;
  }

  int integer = 0;
  const auto int_result =
      std::from_chars(value.data(), value.data() + value.size(), integer);
  if (int_result.ec == std::errc{} &&
      int_result.ptr == value.data() + value.size()) {
    return integer;
  }

  double real = 0.0;
  const auto real_result =
      std::from_chars(value.data(), value.data() + value.size(), real);
  if (real_result.ec == std::errc{} &&
      real_result.ptr == value.data() + value.size()) {
    return real;
  }

  return value;
}

std::size_t parse_size(const std::string &value) {
  int parsed = 0;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
      parsed <= 0) {
    throw std::invalid_argument("expected positive integer");
  }
  return static_cast<std::size_t>(parsed);
}

void apply_key_value(AppConfig &config, Section section,
                     FilterConfig *current_filter, const std::string &key,
                     const std::string &value) {
  switch (section) {
  case Section::Capture:
    if (key == "type") {
      config.capture.type = value;
    } else if (key == "address") {
      config.capture.address = value;
    }
    return;
  case Section::Gpu:
    if (key == "backend") {
      config.gpu.backend = value;
    }
    return;
  case Section::Pipeline:
    if (key == "threads") {
      config.pipeline.threads = value;
    } else if (key == "queue_size") {
      config.pipeline.queue_size = parse_size(value);
    }
    return;
  case Section::Ai:
    if (key == "engine") {
      config.ai.engine = value;
    } else if (key == "model_path") {
      config.ai.model_path = value;
    }
    return;
  case Section::Output:
    if (key == "type") {
      config.output.type = value;
    } else if (key == "device") {
      config.output.device = value;
    } else if (key == "pixel_format") {
      config.output.pixel_format = value;
    } else if (key == "width") {
      config.output.width = parse_size(value);
    } else if (key == "height") {
      config.output.height = parse_size(value);
    } else if (key == "fps") {
      config.output.fps = parse_size(value);
    }
    return;
  case Section::Filters:
    if (current_filter == nullptr) {
      throw std::invalid_argument("filter key without filter item");
    }
    if (key == "type") {
      current_filter->type = value;
    } else if (key == "enabled") {
      const auto scalar = parse_scalar(value);
      if (!std::holds_alternative<bool>(scalar)) {
        throw std::invalid_argument("filter enabled must be boolean");
      }
      current_filter->enabled = std::get<bool>(scalar);
    } else {
      current_filter->parameters.emplace(key, parse_scalar(value));
    }
    return;
  case Section::None:
    throw std::invalid_argument("key outside section");
  }
}

void validate(const AppConfig &config) {
  if (config.capture.type.empty()) {
    throw std::invalid_argument("capture.type is required");
  }
  if (config.gpu.backend.empty()) {
    throw std::invalid_argument("gpu.backend is required");
  }
  if (config.pipeline.threads.empty()) {
    throw std::invalid_argument("pipeline.threads is required");
  }
  if (config.pipeline.queue_size == 0) {
    throw std::invalid_argument("pipeline.queue_size is required");
  }
  if (config.output.type.empty()) {
    throw std::invalid_argument("output.type is required");
  }
  if (config.output.device.empty()) {
    throw std::invalid_argument("output.device is required");
  }
  if (config.output.width == 0U || config.output.height == 0U ||
      config.output.fps == 0U) {
    throw std::invalid_argument("output width, height, and fps are required");
  }
  for (const auto &filter : config.filters) {
    if (filter.type.empty()) {
      throw std::invalid_argument("filter.type is required");
    }
  }
}

} // namespace

AppConfig ConfigLoader::load_file(const std::filesystem::path &path) const {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open config file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return load_string(buffer.str());
}

AppConfig ConfigLoader::load_string(std::string_view yaml) const {
  AppConfig config;
  Section section = Section::None;
  FilterConfig *current_filter = nullptr;
  std::istringstream input{std::string(yaml)};
  std::string raw_line;

  while (std::getline(input, raw_line)) {
    const auto comment = raw_line.find('#');
    const auto line = trim(raw_line.substr(0, comment));
    if (line.empty()) {
      continue;
    }

    if (!starts_with(line, "- ") && line.back() == ':') {
      const auto section_name =
          trim(std::string_view(line).substr(0, line.size() - 1U));
      current_filter = nullptr;
      if (section_name == "capture") {
        section = Section::Capture;
      } else if (section_name == "gpu") {
        section = Section::Gpu;
      } else if (section_name == "ai") {
        section = Section::Ai;
      } else if (section_name == "pipeline") {
        section = Section::Pipeline;
      } else if (section_name == "filters") {
        section = Section::Filters;
      } else if (section_name == "output") {
        section = Section::Output;
      } else {
        throw std::invalid_argument("unknown config section: " + section_name);
      }
      continue;
    }

    std::string key_value = line;
    if (starts_with(line, "- ")) {
      if (section != Section::Filters) {
        throw std::invalid_argument("list item outside filters section");
      }
      config.filters.push_back(
          FilterConfig{.type = {}, .enabled = true, .parameters = {}});
      current_filter = &config.filters.back();
      key_value = trim(std::string_view(line).substr(2));
    }

    const auto [key, value] = split_key_value(key_value);
    apply_key_value(config, section, current_filter, key, value);
  }

  validate(config);
  return config;
}

} // namespace lmp::config
