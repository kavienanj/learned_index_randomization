#pragma once
// Read the plain-text input files produced by each experiment's
// input/generate_inputs.py: one integer per line. Keeping input generation
// in Python and consumption in C++ means every language/tool touching an
// experiment sees byte-identical data for a given seed (docs/REPRODUCIBILITY.md #6).

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rndbench {

inline std::vector<int64_t> ReadIntFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("rndbench::ReadIntFile: could not open " + path +
                              " -- did you run input/generate_inputs.py?");
  }
  std::vector<int64_t> values;
  int64_t v;
  while (in >> v) values.push_back(v);
  return values;
}

}  // namespace rndbench
