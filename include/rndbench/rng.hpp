#pragma once
// Seeded RNG wrapper(s) used by every experiment. The point of this file is
// entirely to make one rule easy to follow: no unseeded randomness anywhere
// near a measured region. See docs/REPRODUCIBILITY.md #1-#2.
//
// Templated on the underlying engine so the *same* algorithm code can be
// instantiated against two different generators -- that's the whole
// mechanism behind the "same Monte Carlo algorithm, different RNG" style of
// experiment (docs/EXPERIMENT_DESIGN.md), e.g.
// experiments/monte_carlo_pi. `Rng` (mt19937_64) is the default,
// good-quality choice for everything else -- reach for a different engine
// only when the RNG itself is the thing under study.

#include <cstdint>
#include <random>
#include <string>

namespace rndbench {

namespace detail {
// Human-readable name per engine, so a run's output can log which
// generator produced it without every experiment hand-rolling this.
template <typename Engine>
struct EngineName {
  static constexpr const char* value = "unknown_engine";
};
template <>
struct EngineName<std::mt19937_64> {
  static constexpr const char* value = "mt19937_64";
};
template <>
struct EngineName<std::minstd_rand> {
  // Park-Miller / Lehmer minimal-standard LCG -- included specifically for
  // its well-documented weak 2D/3D equidistribution ("Marsaglia's planes"),
  // to make that failure mode visible experimentally rather than just
  // asserted. See experiments/monte_carlo_pi/README.md.
  static constexpr const char* value = "minstd_rand (LCG)";
};
}  // namespace detail

template <typename Engine>
class BasicRng {
 public:
  explicit BasicRng(uint64_t seed) : seed_(seed), engine_(static_cast<typename Engine::result_type>(seed)) {}

  // Mint a fresh, unpredictable top-level seed. Call this exactly once per
  // *experiment design* (to decide what seeds to sweep), never inside a
  // measured loop -- and always log the result.
  static uint64_t FreshSeed() {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ rd();
  }

  static constexpr const char* EngineName() { return detail::EngineName<Engine>::value; }

  uint64_t seed() const { return seed_; }

  // Uniform integer in [lo, hi], inclusive.
  int64_t UniformInt(int64_t lo, int64_t hi) {
    std::uniform_int_distribution<int64_t> dist(lo, hi);
    return dist(engine_);
  }

  double UniformReal(double lo = 0.0, double hi = 1.0) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(engine_);
  }

  // Bernoulli trial with given success probability -- e.g. skip-list coin
  // flips, randomized load-balancing decisions.
  bool Coin(double p = 0.5) {
    std::bernoulli_distribution dist(p);
    return dist(engine_);
  }

  Engine& engine() { return engine_; }

 private:
  uint64_t seed_;
  Engine engine_;
};

// Default engine for anything where the RNG itself isn't the variable being
// studied (randomized pivots, coin flips, sampling, ...): mt19937_64, not
// random_device/rand(), so a given seed reproduces bit-identical decisions
// on any machine, any run, forever.
using Rng = BasicRng<std::mt19937_64>;

// A second, deliberately weaker engine (32-bit Lehmer LCG) for experiments
// that compare RNGs against each other rather than comparing a
// deterministic vs. randomized algorithm -- see experiments/monte_carlo_pi.
using LcgRng = BasicRng<std::minstd_rand>;

}  // namespace rndbench
