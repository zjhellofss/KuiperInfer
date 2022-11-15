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
  kParameterMissingUnknown = -1,
  kParameterMissingStride = 0,
  kParameterMissingPadding = 1,
  kParameterMissingKernel = 2,
  kParameterMissingUseBias = 3,
  kParameterMissingInChannel = 4,
  kParameterMissingOutChannel = 5,
  kParameterMissingWeightSize = 6,
  kParameterMissingBiasSize = 7,
  kParameterMissingParamsSize = 8,
  kParameterMissingAttrBias = 9,
  kParameterMissingAttrWeight = 10,
  kParameterParseSuccess = 11
};
}
#endif //KUIPER_COURSE_INCLUDE_COMMON_HPP_
