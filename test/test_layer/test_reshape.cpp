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

// Created by fss on 22-12-21.
#include <gtest/gtest.h>
#include "data/tensor.hpp"

TEST(test_reshape, reshape1) {
  using namespace kuiper_infer;
  Tensor<float> tensor1(3, 224, 224);
  tensor1.RandN();
  Tensor<float> tensor2 = tensor1;
  tensor1.Reshape({224, 224, 3});

  const auto& raw_shapes = tensor1.raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 3);
  ASSERT_EQ(raw_shapes.at(0), 224);
  ASSERT_EQ(raw_shapes.at(1), 224);
  ASSERT_EQ(raw_shapes.at(2), 3);

  ASSERT_EQ(tensor1.size(), tensor2.size());
  uint32_t size = tensor1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor1.index(i), tensor2.index(i));
  }
}

TEST(test_reshape, reshape2) {
  using namespace kuiper_infer;
  Tensor<float> tensor1(3, 224, 224);
  tensor1.RandN();
  Tensor<float> tensor2 = tensor1;
  tensor1.Reshape({672, 224});

  const auto& raw_shapes = tensor1.raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 2);
  ASSERT_EQ(raw_shapes.at(0), 672);
  ASSERT_EQ(raw_shapes.at(1), 224);

  ASSERT_EQ(tensor1.size(), tensor2.size());
  uint32_t size = tensor1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor1.index(i), tensor2.index(i));
  }
}

TEST(test_reshape, reshape3) {
  using namespace kuiper_infer;
  Tensor<float> tensor1(3, 224, 224);
  tensor1.RandN();
  Tensor<float> tensor2 = tensor1;
  tensor1.Reshape({150528});

  const auto& raw_shapes = tensor1.raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 1);
  ASSERT_EQ(raw_shapes.at(0), 150528);

  ASSERT_EQ(tensor1.size(), tensor2.size());
  uint32_t size = tensor1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor1.index(i), tensor2.index(i));
  }
}
