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

// Created by fss on 23-1-22.

#include <gtest/gtest.h>
#include "layer/abstract/layer_factory.hpp"

using namespace kuiper_infer;
StatusCode TestCreateLayer(const std::shared_ptr<RuntimeOperator>& op,
                           std::shared_ptr<Layer<float>>& layer) {
  layer = std::make_shared<Layer<float>>("test3");
  return StatusCode::kSuccess;
}

TEST(test_layer_factory, init) {
  using namespace kuiper_infer;
  const auto& r1 = LayerRegisterer::Registry();
  const auto& r2 = LayerRegisterer::Registry();
  ASSERT_EQ(r1, r2);
}

TEST(test_layer_factory, registerer) {
  using namespace kuiper_infer;
  const uint32_t size1 = LayerRegisterer::Registry()->size();
  LayerRegisterer::RegisterCreator("test1", TestCreateLayer);
  const uint32_t size2 = LayerRegisterer::Registry()->size();
  LayerRegisterer::RegisterCreator("test2", TestCreateLayer);
  const uint32_t size3 = LayerRegisterer::Registry()->size();

  ASSERT_EQ(size2 - size1, 1);
  ASSERT_EQ(size3 - size1, 2);
}

TEST(test_layer_factory, create) {
  using namespace kuiper_infer;
  LayerRegisterer::RegisterCreator("test3", TestCreateLayer);
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "test3";
  std::shared_ptr<Layer<float>> layer = LayerRegisterer::CreateLayer(op);
  ASSERT_EQ(layer->layer_name(), "test3");
}