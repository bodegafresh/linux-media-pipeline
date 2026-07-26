#include "lmp/filters/filter_pipeline.hpp"

namespace lmp::filters {

FilterPipeline
FilterPipeline::from_config(const std::vector<config::FilterConfig> &configs,
                            const FilterRegistry &registry) {
  FilterPipeline pipeline;
  for (const auto &config : configs) {
    if (config.enabled) {
      pipeline.add(registry.create(config));
    }
  }
  return pipeline;
}

void FilterPipeline::add(std::unique_ptr<IVideoFilter> filter) {
  filters_.push_back(std::move(filter));
}

void FilterPipeline::process(frame::Frame &frame) const {
  for (const auto &filter : filters_) {
    filter->process(frame);
  }
}

std::size_t FilterPipeline::size() const noexcept { return filters_.size(); }

} // namespace lmp::filters
