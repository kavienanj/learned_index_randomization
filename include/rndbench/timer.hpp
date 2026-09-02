#pragma once
// Lightweight monotonic stopwatch for ad-hoc/manual timing (e.g. inside a
// correctness test, or a script that isn't a Google Benchmark harness).
// Inside an actual benchmark, prefer Google Benchmark's own timing --
// state.PauseTiming()/ResumeTiming() around setup you don't want measured --
// rather than this class, so results stay comparable across experiments.

#include <chrono>

namespace rndbench {

class Timer {
 public:
  Timer() : start_(Clock::now()) {}

  void Reset() { start_ = Clock::now(); }

  double ElapsedSeconds() const {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }

  double ElapsedMillis() const { return ElapsedSeconds() * 1e3; }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

}  // namespace rndbench
