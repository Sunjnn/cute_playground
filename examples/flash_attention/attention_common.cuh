#ifndef EXAMPLES_FLASH_ATTENTION_ATTENTION_COMMON_CUH
#define EXAMPLES_FLASH_ATTENTION_ATTENTION_COMMON_CUH

#include <cmath>
#include <limits>

namespace attention {

// Finite scale alone does not guarantee finite fp32 logits. Bound against any finite fp16 Q/K,
// with a factor-of-two margin for dot-product rounding, so stable softmax never sees +inf - +inf.
inline bool valid_scale(int k, float scale) {
  if (k <= 0 || !std::isfinite(scale)) {
    return false;
  }
  constexpr double kMaxHalf = 65504.0;
  const auto maxScore = static_cast<double>(k) * kMaxHalf * kMaxHalf;
  return std::fabs(static_cast<double>(scale)) <=
         static_cast<double>(std::numeric_limits<float>::max()) / (2.0 * maxScore);
}

} // namespace attention

#endif
