//
// Created by fss on 22-12-26.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../source/layer/details/cat.hpp"

TEST(test_layer, cat1) {
  using namespace kuiper_infer;
  int input_size = 4;
  int batch_size = 2;
  int in_channels = 6;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(in_channels, 32, 32);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs(batch_size);
  CatLayer cat_layer(1);
  cat_layer.Forward(inputs, outputs);

  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), 12);
  }
}

TEST(test_layer, cat2) {
  using namespace kuiper_infer;
  int input_size = 8;
  int batch_size = 4;
  int in_channels = 32;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(in_channels, 32, 32);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs(batch_size);
  CatLayer cat_layer(1);
  cat_layer.Forward(inputs, outputs);

  for (int i = 0; i < batch_size; ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), 64);
  }
}