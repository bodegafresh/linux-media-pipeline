#pragma once

#include "lmp/config/config.hpp"

#include <filesystem>
#include <string_view>

namespace lmp::config {

class ConfigLoader {
public:
  [[nodiscard]] AppConfig load_file(const std::filesystem::path &path) const;
  [[nodiscard]] AppConfig load_string(std::string_view yaml) const;
};

} // namespace lmp::config
