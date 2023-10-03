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

// Created by fss on 23-1-2.
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/hardsigmoid.hpp"
#include "data/tensor.hpp"

TEST(test_layer, forward_hardsigmoid1) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 224, 512);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  HardSigmoid hardsigmoid_layer;
  const auto status = hardsigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    input_->Transform([](float val) {
      if (val <= -3.f) {
        return 0.f;
      } else if (val >= 3.f) {
        return 1.f;
      } else {
        return val / 6.f + 0.5f;
      }
    });
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_->index(j), input_->index(j));
    }
  }
}

TEST(test_layer, forward_hardsigmoid2) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 32, 128);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  HardSigmoid hardsigmoid_layer;
  const auto status = hardsigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    input_->Transform([](float val) {
      if (val <= -3.f) {
        return 0.f;
      } else if (val >= 3.f) {
        return 1.f;
      } else {
        return val / 6.f + 0.5f;
      }
    });
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_->index(j), input_->index(j));
    }
  }
}

TEST(test_layer, forward_hardsigmoid3) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 16);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  HardSigmoid hardsigmoid_layer;
  const auto status = hardsigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    input_->Transform([](float val) {
      if (val <= -3.f) {
        return 0.f;
      } else if (val >= 3.f) {
        return 1.f;
      } else {
        return val / 6.f + 0.5f;
      }
    });
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_->index(j), input_->index(j));
    }
  }
}