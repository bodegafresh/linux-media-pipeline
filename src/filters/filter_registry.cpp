#include "lmp/filters/filter_registry.hpp"

#include "lmp/filters/box_blur_filter.hpp"
#include "lmp/filters/gaussian_blur_filter.hpp"
#include "lmp/filters/grayscale_filter.hpp"
#include "lmp/filters/identity_filter.hpp"
#include "lmp/filters/negative_filter.hpp"
#include "lmp/filters/sepia_filter.hpp"
#include "lmp/filters/sharpen_filter.hpp"
#include "lmp/filters/sobel_filter.hpp"

#include "filter_config_params.hpp"

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
  registry.register_filter("box_blur", [](const config::FilterConfig &config) {
    return std::make_unique<BoxBlurFilter>(
        detail::radius_parameter(config, 1U));
  });
  registry.register_filter("blur", [](const config::FilterConfig &config) {
    return std::make_unique<BoxBlurFilter>(
        detail::radius_parameter(config, 1U));
  });
  registry.register_filter("gaussian_blur",
                           [](const config::FilterConfig &config) {
                             return std::make_unique<GaussianBlurFilter>(
                                 detail::radius_parameter(config, 1U));
                           });
  registry.register_filter("sharpen", [](const config::FilterConfig &config) {
    const auto amount = detail::double_parameter(config, "amount", 1.0);
    return std::make_unique<SharpenFilter>(amount);
  });
  registry.register_filter("sobel", [](const config::FilterConfig &config) {
    static_cast<void>(config);
    return std::make_unique<SobelFilter>();
  });
  return registry;
}

} // namespace lmp::filters
