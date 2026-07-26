#pragma once

#include "lmp/config/config.hpp"
#include "lmp/filters/video_filter.hpp"

#include <functional>
#include <memory>

namespace lmp::filters {

using FilterFactory =
    std::function<std::unique_ptr<IVideoFilter>(const config::FilterConfig &)>;

} // namespace lmp::filters
