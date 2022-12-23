//
// Created by fss on 22-12-23.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../source/layer/details/linear.hpp"

TEST(test_layer, forward_linear1) {
  using namespace kuiper_infer;
  const uint32_t in_features = 1024;
  const uint32_t out_features = 2048;
  const uint32_t in_dims = 32;
  LinearLayer linear_layer(in_features, out_features, false);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);
  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, 1);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
}


TEST(test_layer, forward_linear2) {
  using namespace kuiper_infer;
  const uint32_t in_features = 3260;
  const uint32_t out_features = 4096;
  const uint32_t in_dims = 128;
  LinearLayer linear_layer(in_features, out_features, false);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);
  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, 1);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
}

TEST(test_layer, forward_linear3) {
  using namespace kuiper_infer;
  const uint32_t in_features = 326;
  const uint32_t out_features = 409;
  const uint32_t in_dims = 1280;
  LinearLayer linear_layer(in_features, out_features, false);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);
  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, 1);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
}

