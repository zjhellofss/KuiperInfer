//
// Created by fss on 22-12-23.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../source/layer/details/linear.hpp"

TEST(test_layer, forward_linear1) {
  using namespace kuiper_infer;
  const uint32_t in_features = 32;
  const uint32_t out_features = 64;
  const uint32_t in_dims = 1280;

  LinearLayer linear_layer(in_features, out_features, false);
  std::vector<float> weights(in_features * out_features, 1.f);
  linear_layer.set_weights(weights);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);
  input->Fill(1.f);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, 1);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), in_features);
    }
  }
}

TEST(test_layer, forward_linear2) {
  using namespace kuiper_infer;
  const uint32_t in_features = 32;
  const uint32_t out_features = 64;
  const uint32_t in_dims = 1280;

  LinearLayer linear_layer(in_features, out_features, false);
  std::vector<float> weights(in_features * out_features, 1.f);
  linear_layer.set_weights(weights);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);
  input->Fill(2.f);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, 1);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), in_features * 2);
    }
  }
}

TEST(test_layer, forward_linear3) {
  using namespace kuiper_infer;
  const uint32_t in_features = 8;
  const uint32_t out_features = 12;
  const uint32_t in_dims = 4;
  LinearLayer linear_layer(in_features, out_features, false);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);
  std::vector<float> weights_raw;

  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }

  linear_layer.set_weights(weights_raw);
  input->Fill(1.f);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, 1);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), 36);
    }
  }
}

TEST(test_layer, forward_linear4) {
  using namespace kuiper_infer;
  const uint32_t in_features = 64;
  const uint32_t out_features = 128;
  const uint32_t in_dims = 4;
  LinearLayer linear_layer(in_features, out_features, false);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);
  std::vector<float> weights_raw;

  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }

  linear_layer.set_weights(weights_raw);
  input->Fill(1.f);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, 1);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), 2080 * 1.f);
    }
  }
}

TEST(test_layer, forward_linear5) {
  using namespace kuiper_infer;
  const uint32_t in_features = 64;
  const uint32_t out_features = 128;
  const uint32_t in_dims = 4;
  LinearLayer linear_layer(in_features, out_features, false);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);
  std::vector<float> weights_raw;

  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }

  linear_layer.set_weights(weights_raw);
  input->Fill(2.f);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, 1);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), 2080 * 2.f);
    }
  }
}


