#pragma once

#include "lmp/filters/filter_factory.hpp"

#include <string>
#include <unordered_map>

namespace lmp::filters {

class FilterRegistry {
public:
  void register_filter(std::string type, FilterFactory factory);

  [[nodiscard]] std::unique_ptr<IVideoFilter>
  create(const config::FilterConfig &config) const;
  [[nodiscard]] bool contains(std::string_view type) const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;

private:
  std::unordered_map<std::string, FilterFactory> factories_;
};

FilterRegistry create_default_registry();

} // namespace lmp::filters
