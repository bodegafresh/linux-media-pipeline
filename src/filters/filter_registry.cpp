#include "lmp/filters/filter_registry.hpp"

#include "lmp/filters/grayscale_filter.hpp"
#include "lmp/filters/identity_filter.hpp"
#include "lmp/filters/negative_filter.hpp"
#include "lmp/filters/sepia_filter.hpp"

#include <stdexcept>

namespace lmp::filters {

void FilterRegistry::register_filter(std::string type, FilterFactory factory) {
  if (type.empty()) {
    throw std::invalid_argument("filter type cannot be empty");
  }
  if (!factory) {
    throw std::invalid_argument("filter factory cannot be empty");
  }
  const auto [_, inserted] =
      factories_.emplace(std::move(type), std::move(factory));
  if (!inserted) {
    throw std::invalid_argument("filter type is already registered");
  }
}

std::unique_ptr<IVideoFilter>
FilterRegistry::create(const config::FilterConfig &config) const {
  const auto found = factories_.find(config.type);
  if (found == factories_.end()) {
    throw std::invalid_argument("unknown filter type: " + config.type);
  }
  return found->second(config);
}

bool FilterRegistry::contains(std::string_view type) const noexcept {
  return factories_.find(std::string(type)) != factories_.end();
}

std::size_t FilterRegistry::size() const noexcept { return factories_.size(); }

FilterRegistry create_default_registry() {
  FilterRegistry registry;
  registry.register_filter("identity", [](const config::FilterConfig &config) {
    static_cast<void>(config);
    return std::make_unique<IdentityFilter>();
  });
  registry.register_filter("grayscale", [](const config::FilterConfig &config) {
    static_cast<void>(config);
    return std::make_unique<GrayscaleFilter>();
  });
  registry.register_filter("negative", [](const config::FilterConfig &config) {
    static_cast<void>(config);
    return std::make_unique<NegativeFilter>();
  });
  registry.register_filter("sepia", [](const config::FilterConfig &config) {
    static_cast<void>(config);
    return std::make_unique<SepiaFilter>();
  });
  return registry;
}

} // namespace lmp::filters
