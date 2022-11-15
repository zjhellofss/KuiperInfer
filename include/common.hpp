//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_INCLUDE_COMMON_HPP_
#define KUIPER_COURSE_INCLUDE_COMMON_HPP_
namespace kuiper_infer {
enum class InferStatus {
  kInferUnknown = -1,
  kInferFailedInputEmpty = 0,
  kInferFailedWeightsOrBiasEmpty = 1,
  kInferFailedInputUnAdapting = 2,
  kInferFailedWeightBiasNoAdapting = 3,
  kInferFailedOutputSizeWrong = 4,
  kInferSuccess = 5,
};

enum class ParseParameterAttrStatus {
  kParameterFailedUnknown = -1,
  kParameterFailedStride = 0,
  kParameterFailedPadding = 1,
  kParameterFailedKernel = 2,
  kParameterFailedUseBias = 3,
  kParameterFailedInChannel = 4,
  kParameterFailedOutChannel = 5,
  kParameterFailedWeightSize = 6,
  kParameterFailedBiasSize = 7,
  kParameterFailedParamsSize = 8,
  kParameterFailedAttrBias = 9,
  kParameterFailedAttrWeight = 10,
  kParameterSuccess = 11
};
}
#endif //KUIPER_COURSE_INCLUDE_COMMON_HPP_
