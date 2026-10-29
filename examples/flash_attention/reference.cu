#include "reference.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

using std::exp;
using std::fabs;
using std::invalid_argument;
using std::isfinite;
using std::max;
using std::numeric_limits;
using std::size_t;
using std::vector;
using thrust::host_vector;

namespace {

constexpr double kNegativeInfinity = -numeric_limits<double>::infinity();

void validate_matrix(int rows, int cols, int ld, size_t count) {
  if (rows <= 0 || cols <= 0 || ld < cols) {
    throw invalid_argument("attention reference requires positive dimensions and valid strides");
  }
  const auto rowCount = static_cast<size_t>(rows - 1);
  const auto colCount = static_cast<size_t>(cols);
  const auto stride = static_cast<size_t>(ld);
  if (rowCount > (numeric_limits<size_t>::max() - colCount) / stride ||
      count < rowCount * stride + colCount) {
    throw invalid_argument("attention reference buffer is too small for its logical matrix");
  }
}

} // namespace

Reference make_reference(
    int m,
    int n,
    int k,
    const host_vector<__half> &hQ,
    int ldQ,
    const host_vector<__half> &hK,
    int ldK,
    const host_vector<__half> &hV,
    int ldV,
    float scale,
    bool causal) {
  validate_matrix(m, k, ldQ, hQ.size());
  validate_matrix(n, k, ldK, hK.size());
  validate_matrix(n, k, ldV, hV.size());
  if (!isfinite(scale)) {
    throw invalid_argument("attention reference requires a finite scale");
  }
  if (causal && m > n) {
    throw invalid_argument("attention reference requires m <= n for causal attention");
  }

  auto ref = Reference();
  ref.output = host_vector<double>(static_cast<size_t>(m) * static_cast<size_t>(k));
  auto qRow = vector<double>(static_cast<size_t>(k));
  auto rowScores = vector<double>(static_cast<size_t>(n));

  for (auto i = 0; i < m; ++i) {
    const auto *query = &hQ[static_cast<size_t>(i) * static_cast<size_t>(ldQ)];
    for (auto t = 0; t < k; ++t) {
      qRow[t] = static_cast<double>(__half2float(query[t]));
    }

    // Only visible keys participate: j <= i + n - m. No full m x n state or masked suffix is
    // needed, and every causal row has at least one key because m <= n was checked above.
    const auto visible = causal ? n - m + 1 + i : n;
    auto rowMax = kNegativeInfinity;
    for (auto j = 0; j < visible; ++j) {
      const auto *key = &hK[static_cast<size_t>(j) * static_cast<size_t>(ldK)];
      auto acc = 0.0;
      for (auto t = 0; t < k; ++t) {
        acc += qRow[t] * static_cast<double>(__half2float(key[t]));
      }
      rowScores[j] = acc * static_cast<double>(scale);
      rowMax = max(rowMax, rowScores[j]);
    }

    // The same row buffer becomes exponentials, then normalized double probabilities. Subtract
    // the maximum before exp() so the reference remains stable even for large finite scores.
    auto rowSum = 0.0;
    for (auto j = 0; j < visible; ++j) {
      rowScores[j] = exp(rowScores[j] - rowMax);
      rowSum += rowScores[j];
    }
    const auto invSum = 1.0 / rowSum;
    for (auto j = 0; j < visible; ++j) {
      rowScores[j] *= invSum;
    }

    auto *output = &ref.output[static_cast<size_t>(i) * static_cast<size_t>(k)];
    for (auto t = 0; t < k; ++t) {
      auto acc = 0.0;
      for (auto j = 0; j < visible; ++j) {
        const auto *value = &hV[static_cast<size_t>(j) * static_cast<size_t>(ldV)];
        acc += rowScores[j] * static_cast<double>(__half2float(value[t]));
      }
      output[t] = acc;
    }
  }
  return ref;
}

double
max_rel_diff(const Reference &ref, int m, int k, const host_vector<__half> &hOut, int ldOut) {
  validate_matrix(m, k, k, ref.output.size());
  validate_matrix(m, k, ldOut, hOut.size());

  auto maxRel = 0.0;
  for (auto i = 0; i < m; ++i) {
    const auto *expectedRow = &ref.output[static_cast<size_t>(i) * static_cast<size_t>(k)];
    const auto *actualRow = &hOut[static_cast<size_t>(i) * static_cast<size_t>(ldOut)];
    for (auto t = 0; t < k; ++t) {
      const auto want = expectedRow[t];
      const auto got = static_cast<double>(__half2float(actualRow[t]));
      if (!isfinite(want) || !isfinite(got)) {
        return numeric_limits<double>::infinity();
      }
      const auto dev = fabs(got - want) / max(fabs(want), 1.0);
      if (!isfinite(dev)) {
        return numeric_limits<double>::infinity();
      }
      maxRel = max(maxRel, dev);
    }
  }
  return maxRel;
}
