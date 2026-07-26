#pragma once

#include "lmp/config/config.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace lmp::filters::detail {

inline int integer_parameter(const config::FilterConfig &config,
                             const std::string &name, int default_value) {
  const auto found = config.parameters.find(name);
  if (found == config.parameters.end()) {
    return default_value;
  }
  if (const auto *value = std::get_if<int>(&found->second)) {
    return *value;
  }
  throw std::invalid_argument(config.type + "." + name + " must be an integer");
}

inline double double_parameter(const config::FilterConfig &config,
                               const std::string &name, double default_value) {
  const auto found = config.parameters.find(name);
  if (found == config.parameters.end()) {
    return default_value;
  }
  if (const auto *value = std::get_if<double>(&found->second)) {
    return *value;
  }
  if (const auto *value = std::get_if<int>(&found->second)) {
    return static_cast<double>(*value);
  }
  throw std::invalid_argument(config.type + "." + name + " must be numeric");
}

inline std::uint32_t radius_parameter(const config::FilterConfig &config,
                                      std::uint32_t default_value) {
  const auto radius =
      integer_parameter(config, "radius", static_cast<int>(default_value));
  if (radius < 1) {
    throw std::invalid_argument(config.type + ".radius must be >= 1");
  }
  return static_cast<std::uint32_t>(radius);
}

inline bool bool_parameter(const config::FilterConfig &config,
                           const std::string &name, bool default_value) {
  const auto found = config.parameters.find(name);
  if (found == config.parameters.end()) {
    return default_value;
  }
  if (const auto *value = std::get_if<bool>(&found->second)) {
    return *value;
  }
  throw std::invalid_argument(config.type + "." + name + " must be boolean");
}

inline std::string string_parameter(const config::FilterConfig &config,
                                    const std::string &name,
                                    const std::string &default_value) {
  const auto found = config.parameters.find(name);
  if (found == config.parameters.end()) {
    return default_value;
  }
  if (const auto *value = std::get_if<std::string>(&found->second)) {
    return *value;
  }
  throw std::invalid_argument(config.type + "." + name + " must be a string");
}

inline std::uint32_t coordinate_parameter(const config::FilterConfig &config,
                                          const std::string &name,
                                          std::uint32_t default_value) {
  const auto value =
      integer_parameter(config, name, static_cast<int>(default_value));
  if (value < 0) {
    throw std::invalid_argument(config.type + "." + name + " must be >= 0");
  }
  return static_cast<std::uint32_t>(value);
}

} // namespace lmp::filters::detail
