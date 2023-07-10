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

// Created by fss on 22-11-28.

#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_PARAMETER_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_PARAMETER_HPP_
#include <string>
#include <vector>
#include "status_code.hpp"

namespace kuiper_infer {
/**
 * 计算节点中的参数信息，参数一共可以分为如下的几类
 * 1.int
 * 2.float
 * 3.string
 * 4.bool
 * 5.int array
 * 6.string array
 * 7.float array
 */
struct RuntimeParameter {  /// 计算节点中的参数信息
  virtual ~RuntimeParameter() = default;

  explicit RuntimeParameter(
      RuntimeParameterType type = RuntimeParameterType::kParameterUnknown)
      : type(type) {}
  RuntimeParameterType type = RuntimeParameterType::kParameterUnknown;
};

struct RuntimeParameterInt : public RuntimeParameter {
  RuntimeParameterInt()
      : RuntimeParameter(RuntimeParameterType::kParameterInt) {}
  int value = 0;
};

struct RuntimeParameterFloat : public RuntimeParameter {
  RuntimeParameterFloat()
      : RuntimeParameter(RuntimeParameterType::kParameterFloat) {}
  float value = 0.f;
};

struct RuntimeParameterString : public RuntimeParameter {
  RuntimeParameterString()
      : RuntimeParameter(RuntimeParameterType::kParameterString) {}
  std::string value;
};

struct RuntimeParameterIntArray : public RuntimeParameter {
  RuntimeParameterIntArray()
      : RuntimeParameter(RuntimeParameterType::kParameterIntArray) {}
  std::vector<int> value;
};

struct RuntimeParameterFloatArray : public RuntimeParameter {
  RuntimeParameterFloatArray()
      : RuntimeParameter(RuntimeParameterType::kParameterFloatArray) {}
  std::vector<float> value;
};

struct RuntimeParameterStringArray : public RuntimeParameter {
  RuntimeParameterStringArray()
      : RuntimeParameter(RuntimeParameterType::kParameterStringArray) {}
  std::vector<std::string> value;
};

struct RuntimeParameterBool : public RuntimeParameter {
  RuntimeParameterBool()
      : RuntimeParameter(RuntimeParameterType::kParameterBool) {}
  bool value = false;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_PARAMETER_HPP_
