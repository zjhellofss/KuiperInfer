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

// Created by fss on 22-12-18.

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/sigmoid.hpp"
#include "data/tensor.hpp"

TEST(test_layer, forward_sigmoid1) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 4);
  input->index(0) = 1.f;
  input->index(1) = 2.f;
  input->index(2) = 3.f;
  input->index(3) = 4.f;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  std::shared_ptr<Tensor<float>> output1 = std::make_shared<Tensor<float>>(1, 1, 4);
  outputs.push_back(output1);

  SigmoidLayer sigmoid_layer;
  const auto status = sigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_LE(std::abs(outputs.front()->index(0) - 0.7310585786300049f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(1) - 0.8807970779778823f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(2) - 0.9525741268224334f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(3) - 0.9820137900379085f), 1e-6);
}

TEST(test_layer, forward_sigmoid2) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 4);
  input->index(0) = 11.f;
  input->index(1) = 22.f;
  input->index(2) = 33.f;
  input->index(3) = 44.f;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  std::shared_ptr<Tensor<float>> output1 = std::make_shared<Tensor<float>>(1, 1, 4);
  outputs.push_back(output1);

  SigmoidLayer sigmoid_layer;
  const auto status = sigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_LE(std::abs(outputs.front()->index(0) - 0.999983298578152f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(1) - 0.9999999997210531f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(2) - 0.9999999999999953f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(3) - 1.0f), 1e-6);
}

TEST(test_layer, forward_sigmoid3) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 224, 512);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  SigmoidLayer sigmoid_layer;
  const auto status = sigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(std::abs(output_->index(j) - 1.f / (1 + std::exp(-input_->index(j)))), 1e-6f);
    }
  }
}

TEST(test_layer, forward_sigmoid4) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 32, 128);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  SigmoidLayer sigmoid_layer;
  const auto status = sigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(std::abs(output_->index(j) - 1.f / (1 + std::exp(-input_->index(j)))), 1e-6f);
    }
  }
}

TEST(test_layer, forward_sigmoid5) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 125);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  SigmoidLayer sigmoid_layer;
  const auto status = sigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(std::abs(output_->index(j) - 1.f / (1 + std::exp(-input_->index(j)))), 1e-6f);
    }
  }
}