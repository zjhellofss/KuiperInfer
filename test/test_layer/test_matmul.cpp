//
// Created by fss on 24-2-10.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include "../../source/layer/details/matmul.hpp"

TEST(test_layer, forward_matmul) {
  using namespace kuiper_infer;
  std::vector<float> weights;
  for (int i = 0; i < 3 * 4; ++i) {
    weights.push_back(float(i));
  }
  LLamaMatmulLayer llama_matmul(3, 4);
  std::shared_ptr<Tensor<float>> weight = std::make_shared<Tensor<float>>(weights.data(), 3, 4);
  llama_matmul.set_weights({weight});

  std::vector<float> inputs;
  for (int i = 0; i < 4 * 5; ++i) {
    inputs.push_back(float(i));
  }

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(inputs.data(), 4, 5);
  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(3, 5);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  llama_matmul.Forward({input}, outputs);
  ASSERT_EQ(output->at(0, 0, 0), 70);
  ASSERT_EQ(output->at(0, 0, 1), 76);
  ASSERT_EQ(output->at(0, 0, 2), 82);
  ASSERT_EQ(output->at(0, 0, 3), 88);
  ASSERT_EQ(output->at(0, 0, 4), 94);

  ASSERT_EQ(output->at(0, 2, 0), 310);
  ASSERT_EQ(output->at(0, 2, 1), 348);
  ASSERT_EQ(output->at(0, 2, 2), 386);
  ASSERT_EQ(output->at(0, 2, 3), 424);
  ASSERT_EQ(output->at(0, 2, 4), 462);
}

TEST(test_layer, forward_matmul2) {
  using namespace kuiper_infer;
  std::vector<float> weights;
  for (int i = 0; i < 3 * 4; ++i) {
    weights.push_back(float(i));
  }
  LLamaMatmulLayer llama_matmul(3, 4);
  std::shared_ptr<Tensor<float>> weight = std::make_shared<Tensor<float>>(weights.data(), 3, 4);
  llama_matmul.set_weights({weight});

  std::vector<float> inputs;
  for (int i = 0; i < 4; ++i) {
    inputs.push_back(float(i));
  }

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(inputs.data(), 4, 1);
  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(3, 1);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  llama_matmul.Forward({input}, outputs);
  ASSERT_EQ(output->at(0, 0, 0), 14);
  ASSERT_EQ(output->at(0, 1, 0), 38);
  ASSERT_EQ(output->at(0, 2, 0), 62);
}

TEST(test_layer, forward_matmul3) {
  using namespace kuiper_infer;
  std::vector<float> weights;
  for (int i = 0; i < 3 * 4; ++i) {
    weights.push_back(float(i));
  }
  LLamaMatmulLayer llama_matmul(1, 4);
  std::shared_ptr<Tensor<float>> weight = std::make_shared<Tensor<float>>(weights.data(), 1, 4);
  llama_matmul.set_weights({weight});

  std::vector<float> inputs;
  for (int i = 0; i < 8; ++i) {
    inputs.push_back(float(i));
  }

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(inputs.data(), 4, 2);
  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, 2);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  llama_matmul.Forward({input}, outputs);
  ASSERT_EQ(output->at(0, 0, 0), 28);
}
