// Correctness tests across multiple seeds and data patterns.

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "learned_index.hpp"
#include "rndbench/rng.hpp"

using rndbench::Rng;
using rndbench::learned_index::FitExact;
using rndbench::learned_index::FitSampled;
using rndbench::learned_index::LinearModel;
using rndbench::learned_index::Lookup;

namespace {

// Generate n sorted, unique keys for test patterns.
std::vector<int64_t> Pattern(const std::string& kind, size_t n) {
  std::vector<int64_t> v;
  if (kind == "sequential") {
    v.resize(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<int64_t>(i);
  } else if (kind == "uniform") {
    Rng rng(42);
    std::set<int64_t> s;
    int64_t cap = std::max<int64_t>(static_cast<int64_t>(n) * 10, 1000);
    while (s.size() < n) s.insert(rng.UniformInt(0, cap));
    v.assign(s.begin(), s.end());
  } else if (kind == "clustered") {
    size_t num_clusters = std::min<size_t>(5, std::max<size_t>(1, n));
    int64_t cluster_gap = 1'000'000;
    size_t idx = 0;
    for (size_t c = 0; c < num_clusters && idx < n; ++c) {
      size_t remaining_clusters = num_clusters - c;
      size_t cluster_size = (n - idx + remaining_clusters - 1) / remaining_clusters;
      for (size_t j = 0; j < cluster_size && idx < n; ++j, ++idx) {
        v.push_back(static_cast<int64_t>(c) * cluster_gap + static_cast<int64_t>(j));
      }
    }
  } else if (kind == "exponential") {
    // Growing gaps (quadratic spacing).
    int64_t key = 0;
    for (size_t i = 0; i < n; ++i) {
      v.push_back(key);
      key += static_cast<int64_t>(i) + 1;
    }
  } else if (kind == "adversarial") {
    // Bulk plus extreme outliers.
    size_t poison = std::min<size_t>(2, n);
    size_t bulk_n = n - poison;
    for (size_t i = 0; i < bulk_n; ++i) v.push_back(static_cast<int64_t>(i) * 2);
    int64_t poison_base = static_cast<int64_t>(bulk_n) * 2 + 1'000'000;
    for (size_t i = 0; i < poison; ++i) {
      v.push_back(poison_base + static_cast<int64_t>(i) * 1000);
    }
  }
  return v;
}

const std::vector<std::string> kPatterns = {"sequential", "uniform", "clustered", "exponential", "adversarial"};
const std::vector<size_t> kSizes = {0, 1, 2, 5, 100, 997};
constexpr size_t kFitSampleSize = 16;

void ExpectEveryKeyFound(const std::vector<int64_t>& keys, const LinearModel& model,
                          const std::string& context) {
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_EQ(Lookup(keys, model, keys[i]), static_cast<int64_t>(i))
        << context << " key=" << keys[i] << " expected_index=" << i;
  }
}

}  // namespace

TEST(LearnedIndex, LookupFindsEveryPresentKeyExactFit) {
  for (const auto& pattern : kPatterns) {
    for (size_t n : kSizes) {
      std::vector<int64_t> keys = Pattern(pattern, n);
      LinearModel model = FitExact(keys);
      ExpectEveryKeyFound(keys, model, "FitExact pattern=" + pattern + " n=" + std::to_string(n));
    }
  }
}

TEST(LearnedIndex, LookupFindsEveryPresentKeySampledFitAcrossManySeeds) {
  for (const auto& pattern : kPatterns) {
    for (size_t n : kSizes) {
      std::vector<int64_t> keys = Pattern(pattern, n);
      for (uint64_t seed = 0; seed < 20; ++seed) {
        Rng rng(seed);
        LinearModel model = FitSampled(keys, kFitSampleSize, rng);
        ExpectEveryKeyFound(keys, model,
                             "FitSampled pattern=" + pattern + " n=" + std::to_string(n) +
                                 " seed=" + std::to_string(seed));
      }
    }
  }
}

TEST(LearnedIndex, LookupReturnsAbsentForMissingKeys) {
  std::vector<int64_t> keys = {10, 20, 30};
  LinearModel model = FitExact(keys);
  EXPECT_EQ(Lookup(keys, model, 5), -1);    // below min
  EXPECT_EQ(Lookup(keys, model, 35), -1);   // above max
  EXPECT_EQ(Lookup(keys, model, 15), -1);   // gap
  EXPECT_EQ(Lookup(keys, model, 25), -1);   // gap
}

TEST(LearnedIndex, EdgeCasesNoCrash) {
  std::vector<int64_t> empty;
  LinearModel exact_empty = FitExact(empty);
  EXPECT_EQ(Lookup(empty, exact_empty, 42), -1);

  Rng rng(0);
  LinearModel sampled_empty = FitSampled(empty, kFitSampleSize, rng);
  EXPECT_EQ(Lookup(empty, sampled_empty, 42), -1);

  std::vector<int64_t> single = {7};
  LinearModel exact_single = FitExact(single);
  EXPECT_EQ(Lookup(single, exact_single, 7), 0);
  EXPECT_EQ(Lookup(single, exact_single, 8), -1);

  Rng rng2(0);
  LinearModel sampled_single = FitSampled(single, kFitSampleSize, rng2);
  EXPECT_EQ(Lookup(single, sampled_single, 7), 0);
}

TEST(LearnedIndex, FitSampledSampleSizeAtOrAboveN) {
  std::vector<int64_t> keys = Pattern("uniform", 50);
  Rng rng_eq(1), rng_gt(2);
  LinearModel model_eq = FitSampled(keys, keys.size(), rng_eq);
  LinearModel model_gt = FitSampled(keys, keys.size() * 10, rng_gt);
  ExpectEveryKeyFound(keys, model_eq, "sample_size == n");
  ExpectEveryKeyFound(keys, model_gt, "sample_size > n");
}

// Lookup must be correct regardless of model quality.
TEST(LearnedIndex, LookupCorrectRegardlessOfModelQuality) {
  std::vector<int64_t> keys = Pattern("clustered", 200);

  LinearModel zero_model{0.0, 0.0};
  ExpectEveryKeyFound(keys, zero_model, "zero model");

  LinearModel wild_model{-3.7, 1e9};
  ExpectEveryKeyFound(keys, wild_model, "wild model");

  // Model from unrelated pattern must still maintain correctness.
  std::vector<int64_t> unrelated = Pattern("sequential", 5);
  LinearModel foreign_model = FitExact(unrelated);
  ExpectEveryKeyFound(keys, foreign_model, "foreign model");
}

// Adversarial pattern must maintain correctness.
TEST(LearnedIndex, AdversarialPatternStaysCorrect) {
  std::vector<int64_t> keys = Pattern("adversarial", 300);
  LinearModel exact_model = FitExact(keys);
  ExpectEveryKeyFound(keys, exact_model, "adversarial FitExact");

  for (uint64_t seed = 0; seed < 20; ++seed) {
    Rng rng(seed);
    LinearModel sampled_model = FitSampled(keys, kFitSampleSize, rng);
    ExpectEveryKeyFound(keys, sampled_model, "adversarial FitSampled seed=" + std::to_string(seed));
  }
}
