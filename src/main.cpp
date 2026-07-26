#include "lmp/version.hpp"

#include <iostream>

int main() {
  std::cout << "linux-media-pipeline " << lmp::version_string() << '\n';
  return 0;
}
