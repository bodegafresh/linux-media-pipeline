#include "lmp/config/config_loader.hpp"
#include "lmp/filters/filter_pipeline.hpp"
#include "lmp/filters/filter_registry.hpp"
#include "lmp/version.hpp"

#include <iostream>

int main() {
  const lmp::config::ConfigLoader loader;
  const auto config = loader.load_file("config/default.yaml");
  const auto registry = lmp::filters::create_default_registry();
  const auto pipeline =
      lmp::filters::FilterPipeline::from_config(config.filters, registry);

  std::cout << "linux-media-pipeline " << lmp::version_string()
            << " filters=" << pipeline.size() << '\n';
  return 0;
}
