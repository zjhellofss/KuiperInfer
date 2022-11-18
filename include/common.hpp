//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_INCLUDE_COMMON_HPP_
#define KUIPER_COURSE_INCLUDE_COMMON_HPP_
namespace kuiper_infer {
enum class InferStatus {
  kInferUnknown = -1,
  kInferFailedInputEmpty = 1,
  kInferFailedWeightParameterError = 2,
  kInferFailedBiasParameterError = 3,
  kInferFailedStrideParameterError = 4,
  kInferFailedDimensionParameterError = 5,
  kInferFailedChannelParameterError = 6,

  kInferFailedOutputSizeError = 7,
  kInferSuccess = 0,
};

enum class ParseParameterAttrStatus {
  kParameterMissingUnknown = -1,
  kParameterMissingStride = 1,
  kParameterMissingPadding = 2,
  kParameterMissingKernel = 3,
  kParameterMissingUseBias = 4,
  kParameterMissingInChannel = 5,
  kParameterMissingOutChannel = 6,

  kParameterMissingEps = 7,
  kParameterMissingNumFeatures = 8,
  kParameterMissingDim = 9,

  kAttrMissingBias =10,
  kAttrMissingWeight = 11,
  kAttrMissingRunningMean = 12,
  kAttrMissingRunningVar = 13,

  kParameterAttrParseSuccess = 0
};
}
#endif //KUIPER_COURSE_INCLUDE_COMMON_HPP_
