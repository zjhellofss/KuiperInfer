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

// Created by yizhu on 2023/2/5.

#include <gtest/gtest.h>
#include "runtime/runtime_parameter.hpp"

TEST(test_runtime, runtime_param1) {
  using namespace kuiper_infer;
  RuntimeParameter* param = new RuntimeParameterInt;
  ASSERT_EQ(param->type, RuntimeParameterType::kParameterInt);
}

TEST(test_runtime, runtime_param2) {
  using namespace kuiper_infer;
  RuntimeParameter* param = new RuntimeParameterInt;
  ASSERT_EQ(param->type, RuntimeParameterType::kParameterInt);
  ASSERT_EQ(dynamic_cast<RuntimeParameterFloat*>(param), nullptr);
  ASSERT_NE(dynamic_cast<RuntimeParameterInt*>(param), nullptr);
}