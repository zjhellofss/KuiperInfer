//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_INCLUDE_COMMON_HPP_
#define KUIPER_COURSE_INCLUDE_COMMON_HPP_
namespace kuiper_infer {

enum class RuntimeDataType {
  kTypeNull = 0,
  kTypeFloat32 = 1,
  kTypeFloat64 = 2,
  kTypeFloat16 = 3,
  kTypeInt32 = 4,
  kTypeInt64 = 5,
  kTypeInt16 = 6,
  kTypeInt8 = 7,
  kTypeUInt8 = 8,
};

enum class RuntimeParameterType {
  kParameterNull = 0,
  kParameterInt = 1,
  kParameterFloat = 2,
  kParameterString = 3,
  kParameterIntArray = 4,
  kParameterFloatArray = 5,
  kParameterStringArray = 6,
  kParameterBool = 7,
};

enum class InferStatus {
  kInferUnknown = -1,
  kInferFailedInputEmpty = 1,
  kInferFailedWeightParameterError = 2,
  kInferFailedBiasParameterError = 3,
  kInferFailedStrideParameterError = 4,
  kInferFailedDimensionParameterError = 5,
  kInferFailedChannelParameterError = 6,

  kInferFailedOutputSizeError = 7,
  kInferFailedOperationUnknown = 8,
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


  kAttrMissingBias = 21,
  kAttrMissingWeight = 22,
  kAttrMissingRunningMean = 23,
  kAttrMissingRunningVar = 24,

  kParameterAttrParseSuccess = 0
};
}
#endif //KUIPER_COURSE_INCLUDE_COMMON_HPP_
