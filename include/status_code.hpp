//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_INFER_INCLUDE_COMMON_HPP_
#define KUIPER_INFER_INCLUDE_COMMON_HPP_
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
  kInferSuccess = 0,

  kInferFailedInputEmpty = 1,
  kInferFailedWeightParameterError = 2,
  kInferFailedBiasParameterError = 3,
  kInferFailedStrideParameterError = 4,
  kInferFailedDimensionParameterError = 5,
  kInferFailedInputOutSizeMatchError = 6,

  kInferFailedOutputSizeError = 7,
  kInferFailedShapeParameterError = 9,
  kInferFailedChannelParameterError = 10,
  kInferFailedOutputEmpty = 11,

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
  kParameterMissingDilation = 16,
  kParameterMissingPaddingMode = 16,

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
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_COMMON_HPP_
