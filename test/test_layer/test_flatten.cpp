//
// Created by fss on 22-11-24.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/flatten.hpp"
#include "data/tensor.hpp"

TEST(test_layer, forward_flatten_layer1) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  inputs.push_back(input);

  FlattenLayer flatten_layer(1, 3);
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);

  const auto& shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24 * 32);
  ASSERT_EQ(shapes.at(2), 1);

  const auto& raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 1);
  ASSERT_EQ(raw_shapes.front(), 8 * 24 * 32);

  uint32_t size1 = inputs.front()->size();
  uint32_t size2 = outputs.front()->size();
  ASSERT_EQ(size1, size2);
}

TEST(test_layer, forward_flatten_layer2) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  inputs.push_back(input);

  FlattenLayer flatten_layer(1, -1);  // 1 3 -> 0 2
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);

  const auto& shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24 * 32);
  ASSERT_EQ(shapes.at(2), 1);

  const auto& raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 1);

  ASSERT_EQ(raw_shapes.front(), 8 * 24 * 32);

  uint32_t size1 = inputs.front()->size();
  uint32_t size2 = outputs.front()->size();
  ASSERT_EQ(size1, size2);
}

TEST(test_layer, forward_flatten_layer3) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  inputs.push_back(input);

  FlattenLayer flatten_layer(1, 2);  // 0 1
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ;
  ASSERT_EQ(outputs.size(), 1);

  const auto& shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24);
  ASSERT_EQ(shapes.at(2), 32);

  const auto& raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 2);

  ASSERT_EQ(raw_shapes.at(0), 8 * 24);
  ASSERT_EQ(raw_shapes.at(1), 32);

  uint32_t size1 = inputs.front()->size();
  uint32_t size2 = outputs.front()->size();
  ASSERT_EQ(size1, size2);
}

TEST(test_layer, forward_flatten_layer4) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  inputs.push_back(input);

  FlattenLayer flatten_layer(1, -2);  // 0 1
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_EQ(outputs.size(), 1);

  const auto& shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24);
  ASSERT_EQ(shapes.at(2), 32);

  const auto& raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 2);

  ASSERT_EQ(raw_shapes.at(0), 8 * 24);
  ASSERT_EQ(raw_shapes.at(1), 32);

  uint32_t size1 = inputs.front()->size();
  uint32_t size2 = outputs.front()->size();
  ASSERT_EQ(size1, size2);
}

TEST(test_layer, forward_flatten_layer5) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  inputs.push_back(input);

  FlattenLayer flatten_layer(2, 3);  // 1 2
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_EQ(outputs.size(), 1);

  const auto& shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8);
  ASSERT_EQ(shapes.at(2), 24 * 32);

  const auto& raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 2);

  ASSERT_EQ(raw_shapes.at(0), 8);
  ASSERT_EQ(raw_shapes.at(1), 24 * 32);

  uint32_t size1 = inputs.front()->size();
  uint32_t size2 = outputs.front()->size();
  ASSERT_EQ(size1, size2);
}

TEST(test_layer, forward_flatten_layer6) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(8, 4, 4);
  std::vector<float> values;
  for (int i = 0; i < 128; ++i) {
    values.push_back(float(i));
  }
  input->Fill(values);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  inputs.push_back(input);
  FlattenLayer flatten_layer(1, 3);  // 1 2
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  const auto& output = outputs.front();
  for (int i = 0; i < 128; ++i) {
    ASSERT_EQ(output->index(i), i);
  }
}

TEST(test_layer, forward_flatten_layer7) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(8, 4, 4);
  std::vector<float> values;
  for (int i = 0; i < 128; ++i) {
    values.push_back(float(i));
  }
  input->Fill(values);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  inputs.push_back(input);
  FlattenLayer flatten_layer(1, 2);  // 0 1
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  const auto& output = outputs.front();

  uint32_t index = 0;
  for (uint32_t row = 0; row < output->rows(); ++row) {
    for (uint32_t col = 0; col < output->cols(); ++col) {
      ASSERT_EQ(output->at(0, row, col), index);
      index += 1;
    }
  }
}

TEST(test_layer, forward_flatten_layer8) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(8, 4, 4);
  std::vector<float> values;
  for (int i = 0; i < 128; ++i) {
    values.push_back(float(i));
  }
  input->Fill(values);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  inputs.push_back(input);
  FlattenLayer flatten_layer(2, 3);  //  1 2
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  const auto& output = outputs.front();

  uint32_t index = 0;
  for (uint32_t row = 0; row < output->rows(); ++row) {
    for (uint32_t col = 0; col < output->cols(); ++col) {
      ASSERT_EQ(output->at(0, row, col), index);
      index += 1;
    }
  }
}