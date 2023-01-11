//
// Created by fss on 22-12-25.
//

#include <gtest/gtest.h>
#include <glog/logging.h>

#include "data/tensor.hpp"
#include "../source/layer/details/silu.hpp"

TEST(test_layer, forward_silu1) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 224, 512);
  input->Rand();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  SiLULayer silu_layer;
  const auto status = silu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(std::abs(output_->index(j) - input_->index(j) / (1 + std::exp(-input_->index(j)))), 1e-6);
    }
  }
}

TEST(test_layer, forward_silu2) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 32, 128);
  input->Rand();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  SiLULayer silu_layer;
  const auto status = silu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(std::abs(output_->index(j) - input_->index(j) / (1 + std::exp(-input_->index(j)))), 1e-6);
    }
  }
}

TEST(test_layer, forward_silu3) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 16);
  input->Rand();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  SiLULayer silu_layer;
  const auto status = silu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(std::abs(output_->index(j) - input_->index(j) / (1 + std::exp(-input_->index(j)))), 1e-6);
    }
  }
}

TEST(test_layer, forward_silu4) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 320, 1);
  input->Rand();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  SiLULayer silu_layer;
  const auto status = silu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(std::abs(output_->index(j) - input_->index(j) / (1 + std::exp(-input_->index(j)))), 1e-6);
    }
  }
}