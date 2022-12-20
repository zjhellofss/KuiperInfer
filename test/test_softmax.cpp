//
// Created by fss on 22-12-19.
//

#include <gtest/gtest.h>
#include <glog/logging.h>

#include "../source/layer/details/softmax.hpp"

TEST(test_layer, forward_softmax1) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = 1.f;
  input->index(1) = 2.f;
  input->index(2) = 3.f;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, 1, 3);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  SoftmaxLayer softmax_layer;
  softmax_layer.Forward(inputs, outputs);

  ASSERT_LE(std::abs(output->index(0) - 0.09003057f), 1e-6);
  ASSERT_LE(std::abs(output->index(1) - 0.24472847f), 1e-6);
  ASSERT_LE(std::abs(output->index(2) - 0.66524096f), 1e-6);

}

TEST(test_layer, forward_softmax2) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = -3.f;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, 1, 3);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  SoftmaxLayer softmax_layer;
  softmax_layer.Forward(inputs, outputs);

  ASSERT_LE(std::abs(output->index(0) - 0.66524096f), 1e-6);
  ASSERT_LE(std::abs(output->index(1) - 0.24472847f), 1e-6);
  ASSERT_LE(std::abs(output->index(2) - 0.09003057f), 1e-6);

}