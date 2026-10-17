#include "reference.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

using std::exp;
using std::fabs;
using std::isnan;
using std::max;
using std::min;
using std::size_t;
using std::vector;
using thrust::host_vector;

namespace {

// Masked positions have to survive being exponentiated and then divided by a finite row sum, so
// they carry -infinity rather than a NaN: exp(-infinity - rowMax) is exactly 0, which is what a
// masked probability should be. Use numeric_limits rather than HUGE_VAL: the UCRT macro expands
// to (double)(float)(1e+300), which nvcc rejects in a constant expression.
constexpr double kMasked = -std::numeric_limits<double>::infinity();

} // namespace

Reference make_reference(
    int m,
    int n,
    int k,
    const host_vector<__half> &hQ,
    int ldQ,
    const host_vector<__half> &hK,
    int ldK,
    float scale,
    bool causal) {
  const auto count = static_cast<size_t>(m) * static_cast<size_t>(n);
  auto ref = Reference();
  ref.scores = host_vector<float>(count);
  ref.probs = host_vector<float>(count);

  // Q's row is widened once and reused across all n columns; K is scanned in its own layout order,
  // so the inner loop walks K's row j contiguously. The m x n x k triple loop is what makes this
  // the slow part of the harness - see main.cu for why verification is optional at large shapes.
  auto qRow = vector<double>(static_cast<size_t>(k));
  auto rowScores = vector<double>(static_cast<size_t>(n));

  for (auto i = 0; i < m; ++i) {
    const auto *query = &hQ[static_cast<size_t>(i) * static_cast<size_t>(ldQ)];
    for (auto t = 0; t < k; ++t) {
      qRow[t] = static_cast<double>(__half2float(query[t]));
    }

    // Column j is visible iff j <= i + n - m, which keeps the last query aligned with the last key
    // when m < n. The clamp is belt-and-braces: main.cu rejects m > n, because a query past the key
    // range would see nothing at all and reduce to 0/0.
    auto visible = n;
    if (causal) {
      visible = min(max(i + n - m + 1, 0), n);
    }

    for (auto j = 0; j < n; ++j) {
      auto acc = 0.0;
      if (j < visible) {
        const auto *key = &hK[static_cast<size_t>(j) * static_cast<size_t>(ldK)];
        for (auto t = 0; t < k; ++t) {
          acc += qRow[t] * static_cast<double>(__half2float(key[t]));
        }
        acc *= static_cast<double>(scale);
      } else {
        acc = kMasked;
      }
      rowScores[j] = acc;
    }

    const auto base = static_cast<size_t>(i) * static_cast<size_t>(n);
    for (auto j = 0; j < n; ++j) {
      ref.scores[base + j] = static_cast<float>(rowScores[j]);
    }

    // Subtracting the row maximum before exponentiating is what keeps exp() away from overflow. It
    // costs an extra walk here, which is the walk the fused kernels are built to avoid.
    auto rowMax = kMasked;
    for (auto j = 0; j < visible; ++j) {
      rowMax = max(rowMax, rowScores[j]);
    }
    auto rowSum = 0.0;
    for (auto j = 0; j < n; ++j) {
      rowScores[j] = exp(rowScores[j] - rowMax);
      rowSum += rowScores[j];
    }
    const auto invSum = 1.0 / rowSum;
    for (auto j = 0; j < n; ++j) {
      ref.probs[base + j] = static_cast<float>(rowScores[j] * invSum);
    }
  }
  return ref;
}

double
max_rel_diff(const Reference &ref, int m, int n, const host_vector<__half> &hOut, int ldOut) {
  auto maxRel = 0.0;
  for (auto i = 0; i < m; ++i) {
    const auto *const expectedRow = &ref.probs[static_cast<size_t>(i) * static_cast<size_t>(n)];
    const auto *const actualRow = &hOut[static_cast<size_t>(i) * static_cast<size_t>(ldOut)];
    for (auto j = 0; j < n; ++j) {
      const auto expected = static_cast<double>(expectedRow[j]);
      const auto actual = static_cast<double>(__half2float(actualRow[j]));
      const auto dev = expected > 0.0 ? fabs(actual - expected) / expected : fabs(actual);
      if (isnan(dev) || dev > maxRel) {
        maxRel = dev;
      }
    }
  }
  return maxRel;
}
