#include "lmp/capture/gopro_udp_source.hpp"
#include "lmp/config/config_loader.hpp"
#include "lmp/filters/filter_pipeline.hpp"
#include "lmp/filters/filter_registry.hpp"
#include "lmp/version.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  const lmp::config::ConfigLoader loader;
  const auto config = loader.load_file("config/default.yaml");
  const auto registry = lmp::filters::create_default_registry();
  const auto pipeline =
      lmp::filters::FilterPipeline::from_config(config.filters, registry);

  auto open_capture = false;
  for (int index = 1; index < argc; ++index) {
    if (std::string_view{argv[index]} == "--open-capture") {
      open_capture = true;
    }
  }

  lmp::capture::GoProUdpSource capture{config.capture.address};
  if (config.capture.type != capture.type()) {
    throw std::runtime_error("unsupported capture type: " +
                             config.capture.type);
  }
  if (open_capture) {
    capture.open();
  }

  std::cout << "linux-media-pipeline " << lmp::version_string()
            << " capture=" << capture.type()
            << " udp=" << capture.endpoint().host << ':'
            << capture.endpoint().port
            << " capture_open=" << (capture.is_open() ? "true" : "false")
            << " filters=" << pipeline.size() << '\n';
  return 0;
}
