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

// Created by fss on 23-1-29.
#include <gtest/gtest.h>
#include "runtime/runtime_attr.hpp"

TEST(test_runtime, attr_weight_data1) {
  using namespace kuiper_infer;
  RuntimeAttribute runtime_attr;
  std::vector<char> weight_data;
  for (int i = 0; i < 32; ++i) {
    weight_data.push_back((char)i);
  }
  runtime_attr.weight_data = weight_data;
  const auto& result_weight_data = runtime_attr.weight_data;
  ASSERT_EQ(result_weight_data.size(), 32);

  for (int i = 0; i < 32; ++i) {
    ASSERT_EQ(weight_data.at(i), (char)i);
  }
}

TEST(test_runtime, attr_weight_data2) {
  using namespace kuiper_infer;
  RuntimeAttribute runtime_attr;
  runtime_attr.type = RuntimeDataType::kTypeFloat32;
  std::vector<char> weight_data;
  for (int i = 0; i < 32; ++i) {
    weight_data.push_back(0);
  }
  runtime_attr.weight_data = weight_data;

  const auto& result_weight_data = runtime_attr.get<float>(true);
  ASSERT_EQ(result_weight_data.size(), 8);
  ASSERT_EQ(runtime_attr.weight_data.size(), 0);

  for (int i = 0; i < 32; ++i) {
    ASSERT_EQ(weight_data.at(i), 0.f);
  }
}

TEST(test_runtime, attr_weight_data3) {
  using namespace kuiper_infer;
  RuntimeAttribute runtime_attr;
  runtime_attr.type = RuntimeDataType::kTypeFloat32;
  std::vector<char> weight_data;
  for (int i = 0; i < 32; ++i) {
    weight_data.push_back(0);
  }
  runtime_attr.weight_data = weight_data;

  const auto& result_weight_data = runtime_attr.get<float>(false);
  ASSERT_EQ(result_weight_data.size(), 8);
  ASSERT_EQ(runtime_attr.weight_data.size(), 32);

  for (int i = 0; i < 32; ++i) {
    ASSERT_EQ(weight_data.at(i), 0.f);
  }
}

TEST(test_runtime, attr_shape) {
  using namespace kuiper_infer;
  RuntimeAttribute runtime_attr;
  runtime_attr.type = RuntimeDataType::kTypeFloat32;
  runtime_attr.shape = std::vector<int>{3, 32, 32};
  ASSERT_EQ(runtime_attr.shape.at(0), 3);
  ASSERT_EQ(runtime_attr.shape.at(1), 32);
  ASSERT_EQ(runtime_attr.shape.at(2), 32);
}
