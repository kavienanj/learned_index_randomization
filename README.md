# Learned Index Experiment

A linear learned index that fits a regression model to predict key positions in a sorted array, then uses exponential search from the prediction to locate keys.

## Variants

- **FitExact (v1)**: Fits linear model using all n keys
- **FitSampled (v2)**: Fits linear model using random sample of m=200 keys

Both share the same query path (exponential + binary search). Correctness is independent of model quality; only query speed depends on prediction accuracy.

## Structure

```
include/
  learned_index.hpp         - LinearModel, FitExact, FitSampled, Lookup
tests/
  test_learned_index.cpp    - Correctness tests
input/
  generate_inputs.py        - Generate synthetic data (5 patterns)
  real_data.py              - Generate real-world data from dictionary
  *.txt                     - Pre-generated test data
CMakeLists.txt              - Build configuration
```

## Data Patterns

- **sequential**: 0, 1, 2, ..., n-1
- **uniform**: Random keys across wide range
- **clustered**: Multiple separated clusters
- **exponential**: Geometrically growing gaps
- **adversarial**: Bulk + far outliers
- **real_wordhash**: Real dictionary words hashed

Sizes: 1,000 | 10,000 | 100,000

## Building & Testing

```bash
mkdir build && cd build
cmake .. -DRNDBENCH_BUILD_TESTS=ON
ctest --verbose
```

## Test Coverage

- FitExact correctness across 5 patterns × 6 sizes
- FitSampled correctness across 20+ seeds
- Missing key detection
- Edge cases (empty, single element)
- Correctness regardless of model quality
- Adversarial pattern stability
