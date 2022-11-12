//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_INCLUDE_COMMON_HPP_
#define KUIPER_COURSE_INCLUDE_COMMON_HPP_
namespace kuiper_infer {
enum class InferStatus {
  kInferUnknown = -1,
  kInferFailedInputEmpty = 0,
  kInferFailedWeightsBiasEmpty = 1,
  kInferFailedInputUnAdapting = 2,
  kInferFailedWeightBiasNoAdapting = 3,
  kInferFailedOutputEmpty = 4,
  kInferSuccess = 5,
};
}
#endif //KUIPER_COURSE_INCLUDE_COMMON_HPP_
