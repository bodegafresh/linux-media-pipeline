#include "lmp/capture/gopro_udp_source.hpp"
#include "lmp/config/config_loader.hpp"
#include "lmp/filters/filter_pipeline.hpp"
#include "lmp/filters/filter_registry.hpp"
#include "lmp/output/v4l2_output.hpp"
#include "lmp/version.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    const lmp::config::ConfigLoader loader;
    const auto config = loader.load_file("config/default.yaml");
    const auto registry = lmp::filters::create_default_registry();
    const auto pipeline =
        lmp::filters::FilterPipeline::from_config(config.filters, registry);

    auto open_capture = false;
    auto check_output = false;
    for (int index = 1; index < argc; ++index) {
      if (std::string_view{argv[index]} == "--open-capture") {
        open_capture = true;
      } else if (std::string_view{argv[index]} == "--check-output") {
        check_output = true;
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

    lmp::output::V4l2Output output{config.output.device};
    if (config.output.type != output.type()) {
      throw std::runtime_error("unsupported output type: " +
                               config.output.type);
    }
    if (check_output) {
      output.open();
    }

    std::cout << "linux-media-pipeline " << lmp::version_string()
              << " capture=" << capture.type()
              << " udp=" << capture.endpoint().host << ':'
              << capture.endpoint().port
              << " capture_open=" << (capture.is_open() ? "true" : "false")
              << " output=" << output.type() << " device=" << output.device()
              << " output_open=" << (output.is_open() ? "true" : "false")
              << " filters=" << pipeline.size() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "linux-media-pipeline: " << error.what() << '\n';
    return 1;
  }
}
