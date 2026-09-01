#pragma once
// Linear learned index: model predicts key position, exponential+binary search finds key.
// Correctness independent of model quality; only speed depends on prediction accuracy.
// Precondition: sorted_keys must be sorted ascending with no duplicates.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "rndbench/rng.hpp"

namespace rndbench::learned_index {

struct LinearModel {
  double slope = 0.0;
  double intercept = 0.0;

  // Predict key position, clamped to [0, n-1].
  int64_t Predict(int64_t key, size_t n) const {
    if (n == 0) return 0;
    double p = std::round(slope * static_cast<double>(key) + intercept);
    double lo = 0.0;
    double hi = static_cast<double>(n - 1);
    p = std::clamp(p, lo, hi);
    return static_cast<int64_t>(p);
  }
};

namespace detail {

// Centered OLS to avoid precision loss on large int64 keys.
inline LinearModel FitOlsCentered(const std::vector<std::pair<int64_t, int64_t>>& points) {
  if (points.empty()) return LinearModel{};

  double sum_x = 0.0, sum_y = 0.0;
  for (const auto& [x, y] : points) {
    sum_x += static_cast<double>(x);
    sum_y += static_cast<double>(y);
  }
  double n = static_cast<double>(points.size());
  double x_bar = sum_x / n;
  double y_bar = sum_y / n;

  double sxy = 0.0, sxx = 0.0;
  for (const auto& [x, y] : points) {
    double dx = static_cast<double>(x) - x_bar;
    double dy = static_cast<double>(y) - y_bar;
    sxy += dx * dy;
    sxx += dx * dx;
  }

  // Degenerate case: fall back to flat model if insufficient spread.
  constexpr double kMinSxx = 1e-9;
  if (sxx < kMinSxx) {
    return LinearModel{0.0, y_bar};
  }
  double slope = sxy / sxx;
  double intercept = y_bar - slope * x_bar;
  return LinearModel{slope, intercept};
}

}  // namespace detail

// v1: OLS over all keys (deterministic).
inline LinearModel FitExact(const std::vector<int64_t>& sorted_keys) {
  std::vector<std::pair<int64_t, int64_t>> points;
  points.reserve(sorted_keys.size());
  for (size_t i = 0; i < sorted_keys.size(); ++i) {
    points.emplace_back(sorted_keys[i], static_cast<int64_t>(i));
  }
  return detail::FitOlsCentered(points);
}

// v2: OLS over random sample of sample_size keys (with replacement).
template <typename RngT>
LinearModel FitSampled(const std::vector<int64_t>& sorted_keys, size_t sample_size, RngT& rng) {
  if (sorted_keys.empty()) return LinearModel{};
  sample_size = std::max<size_t>(1, sample_size);

  std::vector<std::pair<int64_t, int64_t>> points;
  points.reserve(sample_size);
  int64_t max_idx = static_cast<int64_t>(sorted_keys.size()) - 1;
  for (size_t i = 0; i < sample_size; ++i) {
    int64_t idx = rng.UniformInt(0, max_idx);
    points.emplace_back(sorted_keys[static_cast<size_t>(idx)], idx);
  }
  return detail::FitOlsCentered(points);
}

// Exponential search from prediction, then binary search. Returns index or -1.
inline int64_t Lookup(const std::vector<int64_t>& sorted_keys, const LinearModel& model, int64_t key) {
  int64_t n = static_cast<int64_t>(sorted_keys.size());
  if (n == 0) return -1;

  int64_t pred = model.Predict(key, sorted_keys.size());
  if (sorted_keys[static_cast<size_t>(pred)] == key) return pred;

  int64_t lo, hi;
  if (sorted_keys[static_cast<size_t>(pred)] < key) {
    // True position is to the right of the prediction.
    int64_t i = 1;
    while (pred + i < n && sorted_keys[static_cast<size_t>(pred + i)] < key) {
      i *= 2;
    }
    lo = pred + i / 2;
    hi = std::min(pred + i, n - 1);
  } else {
    // True position is to the left of the prediction.
    int64_t i = 1;
    while (pred - i >= 0 && sorted_keys[static_cast<size_t>(pred - i)] > key) {
      i *= 2;
    }
    lo = std::max(pred - i, int64_t{0});
    hi = pred - i / 2;
  }

  auto begin = sorted_keys.begin() + lo;
  auto end = sorted_keys.begin() + hi + 1;
  auto it = std::lower_bound(begin, end, key);
  if (it != end && *it == key) {
    return static_cast<int64_t>(it - sorted_keys.begin());
  }
  return -1;
}

// Mean absolute prediction error across all keys.
inline double MeanAbsoluteError(const std::vector<int64_t>& sorted_keys, const LinearModel& model) {
  if (sorted_keys.empty()) return 0.0;
  double total = 0.0;
  for (size_t i = 0; i < sorted_keys.size(); ++i) {
    int64_t pred = model.Predict(sorted_keys[i], sorted_keys.size());
    total += std::abs(static_cast<double>(pred) - static_cast<double>(i));
  }
  return total / static_cast<double>(sorted_keys.size());
}

}  // namespace rndbench::learned_index
