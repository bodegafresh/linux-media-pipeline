#pragma once

#include "lmp/config/config.hpp"
#include "lmp/filters/filter_registry.hpp"
#include "lmp/frame/frame.hpp"

#include <memory>
#include <vector>

namespace lmp::filters {

class FilterPipeline {
public:
  FilterPipeline() = default;

  static FilterPipeline
  from_config(const std::vector<config::FilterConfig> &configs,
              const FilterRegistry &registry);

  void add(std::unique_ptr<IVideoFilter> filter);
  void process(frame::Frame &frame) const;

  [[nodiscard]] std::size_t size() const noexcept;

private:
  std::vector<std::unique_ptr<IVideoFilter>> filters_;
};

} // namespace lmp::filters
