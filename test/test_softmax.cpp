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

  ASSERT_NEAR(output->index(0), 0.09003057f, 1e-6);
  ASSERT_NEAR(output->index(1), 0.24472847f, 1e-6);
  ASSERT_NEAR(output->index(2), 0.66524096f, 1e-6);
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

  ASSERT_NEAR(output->index(0), 0.66524096f, 1e-6);
  ASSERT_NEAR(output->index(1), 0.24472847f, 1e-6);
  ASSERT_NEAR(output->index(2), 0.09003057f, 1e-6);
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

  ASSERT_NEAR(output->index(0), 0.33333333f, 1e-6);
  ASSERT_NEAR(output->index(1), 0.33333333f, 1e-6);
  ASSERT_NEAR(output->index(2), 0.33333333f, 1e-6);
}

TEST(test_layer, forward_softmax4) {
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

  ASSERT_NEAR(output->index(0), 0.01165623f, 1e-6);
  ASSERT_NEAR(output->index(1), 0.03168492f, 1e-6);
  ASSERT_NEAR(output->index(2), 0.08612854f, 1e-6);
  ASSERT_NEAR(output->index(3), 0.23412166f, 1e-6);
  ASSERT_NEAR(output->index(4), 0.63640865f, 1e-6);
}

TEST(test_layer, forward_softmax5) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 10);
  input->index(0) = 0.f;
  input->index(1) = 1.f;
  input->index(2) = 2.f;
  input->index(3) = 3.f;
  input->index(4) = 4.f;
  input->index(5) = 5.f;
  input->index(6) = 6.f;
  input->index(7) = 7.f;
  input->index(8) = 8.f;
  input->index(9) = 9.f;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, 1, 10);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  SoftmaxLayer softmax_layer;
  const auto status = softmax_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_NEAR(output->index(0), 7.80134161e-05, 1e-6);
  ASSERT_NEAR(output->index(1), 2.12062451e-04, 1e-6);
  ASSERT_NEAR(output->index(2), 5.76445508e-04, 1e-6);
  ASSERT_NEAR(output->index(3), 1.56694135e-03, 1e-6);
  ASSERT_NEAR(output->index(4), 4.25938820e-03, 1e-6);

  ASSERT_NEAR(output->index(5), 1.15782175e-02, 1e-6);
  ASSERT_NEAR(output->index(6), 3.14728583e-02, 1e-6);
  ASSERT_NEAR(output->index(7), 8.55520989e-02, 1e-6);
  ASSERT_NEAR(output->index(8), 2.32554716e-01, 1e-6);
  ASSERT_NEAR(output->index(9), 6.32149258e-01, 1e-6);

}

