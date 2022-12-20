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
  const auto status = softmax_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

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
  const auto status = softmax_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_LE(std::abs(output->index(0) - 0.66524096f), 1e-6);
  ASSERT_LE(std::abs(output->index(1) - 0.24472847f), 1e-6);
  ASSERT_LE(std::abs(output->index(2) - 0.09003057f), 1e-6);
}

TEST(test_layer, forward_softmax3) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = 1.f;
  input->index(1) = 1.f;
  input->index(2) = 1.f;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, 1, 3);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  SoftmaxLayer softmax_layer;
  const auto status = softmax_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_LE(std::abs(output->index(0) - 0.33333333f), 1e-6);
  ASSERT_LE(std::abs(output->index(1) - 0.33333333f), 1e-6);
  ASSERT_LE(std::abs(output->index(2) - 0.33333333f), 1e-6);
}

TEST(test_layer, forward_softmax4) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -3.f;
  input->index(1) = 5.f;
  input->index(2) = 9.f;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, 1, 3);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  SoftmaxLayer softmax_layer;
  const auto status = softmax_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_LE(std::abs(output->index(0) - 6.03366485e-06f), 1e-6);
  ASSERT_LE(std::abs(output->index(1) - 1.79861014e-02f), 1e-6);
  ASSERT_LE(std::abs(output->index(2) - 9.82007865e-01f), 1e-6);
}

TEST(test_layer, forward_softmax5) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 5);
  input->index(0) = 1.f;
  input->index(1) = 2.f;
  input->index(2) = 3.f;
  input->index(3) = 4.f;
  input->index(4) = 5.f;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, 1, 5);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  SoftmaxLayer softmax_layer;
  const auto status = softmax_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_LE(std::abs(output->index(0) - 0.01165623f), 1e-6);
  ASSERT_LE(std::abs(output->index(1) - 0.03168492f), 1e-6);
  ASSERT_LE(std::abs(output->index(2) - 0.08612854f), 1e-6);
  ASSERT_LE(std::abs(output->index(3) - 0.23412166f), 1e-6);
  ASSERT_LE(std::abs(output->index(4) - 0.63640865f), 1e-6);
}