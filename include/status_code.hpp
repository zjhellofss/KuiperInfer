// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-11-12.

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

  kInferInputsEmpty = 1,
  kInferOutputsEmpty = 2,
  kInferParameterError = 3,

  kInferArraySizeMismatch = 4,  // 输入输出张量的数量不一致
  kInferLayerNotImplement = 5,
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
  kParameterMissingPaddingMode = 17,
  kParameterMissingOutputPadding = 18,
  kParameterMissingAlignCorner = 19,

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
