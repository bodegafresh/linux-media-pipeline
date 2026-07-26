#include "lmp/version.hpp"

#include <iostream>
#include <string_view>

namespace {

bool expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  const auto current = lmp::version();
  bool ok = true;
  ok = expect(current.major == 0, "major version") && ok;
  ok = expect(current.minor == 1, "minor version") && ok;
  ok = expect(current.patch == 0, "patch version") && ok;
  ok = expect(lmp::version_string() == "0.1.0", "version string") && ok;
  return ok ? 0 : 1;
}
