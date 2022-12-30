//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_INCLUDE_COMMON_HPP_
#define KUIPER_COURSE_INCLUDE_COMMON_HPP_
namespace kuiper_infer {

enum class RuntimeParameterType {
  kParameterUnknown = 0,
  kParameterBool = 1,
  kParameterInt = 2,

  kParameterFloat = 3,
  kParameterString = 4,
  kParameterIntArray = 5,
  kParameterFloatArray = 6,
  kParameterStringArray = 7,
};

enum class InferStatus {
  kInferUnknown = -1,
  kInferFailedInputEmpty = 1,
  kInferFailedWeightParameterError = 2,
  kInferFailedBiasParameterError = 3,
  kInferFailedStrideParameterError = 4,
  kInferFailedDimensionParameterError = 5,
  kInferFailedChannelParameterError = 6,
  kInferFailedInputOutSizeAdaptingError = 6,

  kInferFailedOutputSizeError = 7,
  kInferFailedOperationUnknown = 8,
  kInferFailedYoloStageNumberError = 9,

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
  kParameterMissingExpr = 10,
  kParameterMissingOutHW = 11,
  kParameterMissingShape = 12,
  kParameterMissingGroups = 13,
  kParameterMissingScale = 14,
  kParameterMissingResizeMode = 15,

  kAttrMissingBias = 21,
  kAttrMissingWeight = 22,
  kAttrMissingRunningMean = 23,
  kAttrMissingRunningVar = 24,
  kAttrMissingOutFeatures = 25,
  kAttrMissingYoloStrides = 26,
  kAttrMissingYoloAnchorGrides = 27,
  kAttrMissingYoloGrides = 28,

  kParameterAttrParseSuccess = 0
};
}
#endif //KUIPER_COURSE_INCLUDE_COMMON_HPP_
