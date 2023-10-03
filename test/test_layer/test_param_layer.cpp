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

// Created by fss on 23-1-23.
#include <gtest/gtest.h>
#include "layer/abstract/param_layer.hpp"

TEST(test_param_layer, set_weight1) {
  using namespace kuiper_infer;
  ParamLayer param_layer("param");
  uint32_t weight_count = 4;
  std::vector<std::shared_ptr<ftensor>> weights;
  for (uint32_t w = 0; w < weight_count; ++w) {
    std::shared_ptr<ftensor> weight = std::make_shared<ftensor>(3, 32, 32);
    weights.push_back(weight);
  }
  param_layer.InitWeightParam(weight_count, 3, 32, 32);
  param_layer.set_weights(weights);
  const auto& weights_ = param_layer.weights();
  ASSERT_EQ(weights.size(), weight_count);

  for (uint32_t w = 0; w < weight_count; ++w) {
    const auto& weight_ = weights_.at(w);
    const auto& weight = weights.at(w);
    ASSERT_EQ(weight->size(), weight_->size());
    for (uint32_t i = 0; i < weight->size(); ++i) {
      ASSERT_EQ(weight->index(i), weight_->index(i));
    }
  }
}

TEST(test_param_layer, set_bias1) {
  using namespace kuiper_infer;
  ParamLayer param_layer("param");
  uint32_t bias_count = 4;
  std::vector<std::shared_ptr<ftensor>> biases;
  for (uint32_t w = 0; w < bias_count; ++w) {
    std::shared_ptr<ftensor> bias = std::make_shared<ftensor>(1, 32, 1);
    biases.push_back(bias);
  }
  param_layer.InitBiasParam(bias_count, 1, 32, 1);
  param_layer.set_bias(biases);
  const auto& biases_ = param_layer.bias();
  ASSERT_EQ(biases.size(), biases_.size());

  for (uint32_t w = 0; w < bias_count; ++w) {
    const auto& bias_ = biases_.at(w);
    const auto& bias = biases.at(w);
    ASSERT_EQ(bias->size(), bias_->size());
    for (uint32_t i = 0; i < bias->size(); ++i) {
      ASSERT_EQ(bias->index(i), bias_->index(i));
    }
  }
}

TEST(test_param_layer, set_weight2) {
  using namespace kuiper_infer;
  ParamLayer param_layer("param");
  uint32_t weight_count = 9;
  std::vector<float> weights;
  for (int i = 0; i < weight_count; ++i) {
    weights.push_back(float(i));
  }
  std::vector<std::shared_ptr<ftensor>> weights_;
  std::shared_ptr<ftensor> weight = std::make_shared<ftensor>(1, 3, 3);
  weights_.push_back(weight);
  param_layer.InitWeightParam(1, 1, 3, 3);
  param_layer.set_weights(weights_);

  param_layer.set_weights(weights);
  weights_ = param_layer.weights();
  for (int i = 0; i < weights_.size(); ++i) {
    const auto weight_ = weights_.at(i);
    int index = 0;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        ASSERT_EQ(weight_->at(0, r, c), index);
        index += 1;
      }
    }
  }
}

TEST(test_param_layer, set_bias2) {
  using namespace kuiper_infer;
  ParamLayer param_layer("param");
  uint32_t weight_count = 9;
  std::vector<float> biases;
  for (int i = 0; i < weight_count; ++i) {
    biases.push_back(float(i));
  }
  std::vector<std::shared_ptr<ftensor>> biases_;
  std::shared_ptr<ftensor> bias = std::make_shared<ftensor>(1, 9, 1);
  biases_.push_back(bias);
  param_layer.InitBiasParam(1, 1, 9, 1);

  param_layer.set_bias(biases_);

  param_layer.set_bias(biases);
  biases_ = param_layer.bias();
  for (int i = 0; i < biases_.size(); ++i) {
    const auto bias_ = biases_.at(i);
    int index = 0;
    for (int r = 0; r < 9; ++r) {
      ASSERT_EQ(bias_->at(0, r, 0), index);
      index += 1;
    }
  }
}

TEST(test_param_layer, init_bias) {
  using namespace kuiper_infer;
  ParamLayer param_layer("param");
  param_layer.InitBiasParam(3, 64, 1, 1);
  const std::vector<sftensor> bias = param_layer.bias();
  ASSERT_EQ(bias.size(), 3);
  for (uint32_t i = 0; i < bias.size(); ++i) {
    ASSERT_EQ(bias.at(i)->channels(), 64);
    ASSERT_EQ(bias.at(i)->rows(), 1);
    ASSERT_EQ(bias.at(i)->cols(), 1);
  }
}

TEST(test_param_layer, init_weight) {
  using namespace kuiper_infer;
  ParamLayer param_layer("param");
  param_layer.InitWeightParam(3, 64, 32, 32);
  const std::vector<sftensor> weight = param_layer.weights();
  ASSERT_EQ(weight.size(), 3);
  for (uint32_t i = 0; i < weight.size(); ++i) {
    ASSERT_EQ(weight.at(i)->channels(), 64);
    ASSERT_EQ(weight.at(i)->rows(), 32);
    ASSERT_EQ(weight.at(i)->cols(), 32);
  }
}