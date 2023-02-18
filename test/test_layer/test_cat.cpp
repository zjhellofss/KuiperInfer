//
// Created by fss on 22-12-26.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../../source/layer/details/cat.hpp"

TEST(test_layer, cat1) {
  using namespace kuiper_infer;
  int input_size = 8;
  int input_channels = 4;

  int output_size = 2;
  int output_channels = 16;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_channels, 32, 32);
    input->Fill(float(i));
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size);
  CatLayer cat_layer(1);
  const auto status = cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), output_channels);
  }

  for (uint32_t i = 0; i < input_size; ++i) {
    sftensor input = inputs.at(i);
    for (uint32_t j = 0; j < input_channels; ++j) {
      const arma::fmat &in_channel = input->at(j);
      uint32_t index = i * input_channels + j;
      uint32_t i_index = index / output_channels;
      uint32_t j_index = index % output_channels;
      const arma::fmat &out_channel = outputs.at(i_index)->at(j_index);
    }
  }
}

TEST(test_layer, cat2) {
  using namespace kuiper_infer;
  int input_size = 16;
  int input_channels = 32;

  int output_size = 2;
  int output_channels = 256;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_channels, 32, 32);
    input->Rand();
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size);
  CatLayer cat_layer(1);
  const auto status = cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), output_channels);
  }

  for (uint32_t i = 0; i < input_size; ++i) {
    sftensor input = inputs.at(i);
    for (uint32_t j = 0; j < input_channels; ++j) {
      const arma::fmat &in_channel = input->at(j);
      uint32_t index = i * input_channels + j;
      uint32_t i_index = index / output_channels;
      uint32_t j_index = index % output_channels;
      const arma::fmat &out_channel = outputs.at(i_index)->at(j_index);
    }
  }
}

TEST(test_layer, cat3) {
  using namespace kuiper_infer;
  int input_size = 16;
  int input_channels = 32;

  int output_size = 1;
  int output_channels = 512;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_channels, 32, 32);
    input->Rand();
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size);
  CatLayer cat_layer(1);
  const auto status = cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), output_channels);
  }

  for (uint32_t i = 0; i < input_size; ++i) {
    sftensor input = inputs.at(i);
    for (uint32_t j = 0; j < input_channels; ++j) {
      const arma::fmat &in_channel = input->at(j);
      uint32_t index = i * input_channels + j;
      uint32_t i_index = index / output_channels;
      uint32_t j_index = index % output_channels;
      const arma::fmat &out_channel = outputs.at(i_index)->at(j_index);
    }
  }
}

TEST(test_layer, cat4) {
  using namespace kuiper_infer;
  int input_size = 320;
  int input_channels = 160;

  int output_size = 160;
  int output_channels = 320;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_channels, 32, 32);
    input->Rand();
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size);
  CatLayer cat_layer(1);
  const auto status = cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), output_channels);
  }

  for (uint32_t i = 0; i < input_size; ++i) {
    sftensor input = inputs.at(i);
    for (uint32_t j = 0; j < input_channels; ++j) {
      const arma::fmat &in_channel = input->at(j);
      uint32_t index = i * input_channels + j;
      uint32_t i_index = index / output_channels;
      uint32_t j_index = index % output_channels;
      const arma::fmat &out_channel = outputs.at(i_index)->at(j_index);
    }
  }
}

