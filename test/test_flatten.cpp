//
// Created by fss on 22-11-24.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../source/layer/details/flatten.hpp"

TEST(test_flatten, flatten1) {
  using namespace kuiper_infer;
  const uint32_t channels = 3;
  const uint32_t rows = 32;
  const uint32_t cols = 32;
  Tensor<float> tensor(channels, rows, cols);
  int value = 0;
  for (uint32_t c = 0; c < channels; ++c) {
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c_ = 0; c_ < cols; ++c_) {
        tensor.at(c, r, c_) = (float) value;
        value += 1;
      }
    }
  }

  tensor.Flatten();
  const uint32_t size = rows * cols * channels;
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor.index(i), i);
  }
  const auto &shapes = tensor.raw_shapes();
  ASSERT_EQ(shapes.size(), 1);
  ASSERT_EQ(shapes.at(0), size);
}

TEST(test_flatten, flatten2) {
  using namespace kuiper_infer;
  const uint32_t channels = 4;
  const uint32_t rows = 320;
  const uint32_t cols = 320;
  Tensor<float> tensor(channels, rows, cols);
  int value = 0;
  for (uint32_t c = 0; c < channels; ++c) {
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c_ = 0; c_ < cols; ++c_) {
        tensor.at(c, r, c_) = (float) value;
        value += 1;
      }
    }
  }

  tensor.Flatten();
  const uint32_t size = rows * cols * channels;
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor.index(i), i);
  }
  const auto &shapes = tensor.raw_shapes();
  ASSERT_EQ(shapes.size(), 1);
  ASSERT_EQ(shapes.at(0), size);
}

TEST(test_layer, forward_flatten_layer1) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  FlattenLayer flatten_layer(1, 3);
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24 * 32);
  ASSERT_EQ(shapes.at(2), 1);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 1);
  ASSERT_EQ(raw_shapes.front(), 8 * 24 * 32);

  uint32_t size1 = inputs.front()->size();
  uint32_t size2 = outputs.front()->size();
  ASSERT_EQ(size1, size2);

  for (uint32_t i = 0; i < size1; ++i) {
    ASSERT_EQ(inputs.front()->index(i), outputs.front()->index(i));
  }
}

TEST(test_layer, forward_flatten_layer2) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  FlattenLayer flatten_layer(1, -1); // 1 3 -> 0 2
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24 * 32);
  ASSERT_EQ(shapes.at(2), 1);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 1);

  ASSERT_EQ(raw_shapes.front(), 8 * 24 * 32);

  uint32_t size1 = inputs.front()->size();
  uint32_t size2 = outputs.front()->size();
  ASSERT_EQ(size1, size2);

  for (uint32_t i = 0; i < size1; ++i) {
    ASSERT_EQ(inputs.front()->index(i), outputs.front()->index(i));
  }

}

TEST(test_layer, forward_flatten_layer3) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  FlattenLayer flatten_layer(1, 2); // 0 1
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);;
  ASSERT_EQ(outputs.size(), 1);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24);
  ASSERT_EQ(shapes.at(2), 32);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 2);

  ASSERT_EQ(raw_shapes.at(0), 8 * 24);
  ASSERT_EQ(raw_shapes.at(1), 32);

  uint32_t size1 = inputs.front()->size();
  uint32_t size2 = outputs.front()->size();
  ASSERT_EQ(size1, size2);

  for (uint32_t i = 0; i < size1; ++i) {
    ASSERT_EQ(inputs.front()->index(i), outputs.front()->index(i));
  }
}

TEST(test_layer, forward_flatten_layer4) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  FlattenLayer flatten_layer(1, -2); // 0 1
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_EQ(outputs.size(), 1);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24);
  ASSERT_EQ(shapes.at(2), 32);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 2);

  ASSERT_EQ(raw_shapes.at(0), 8 * 24);
  ASSERT_EQ(raw_shapes.at(1), 32);

  uint32_t size1 = inputs.front()->size();
  uint32_t size2 = outputs.front()->size();
  ASSERT_EQ(size1, size2);

  for (uint32_t i = 0; i < size1; ++i) {
    ASSERT_EQ(inputs.front()->index(i), outputs.front()->index(i));
  }
}

TEST(test_layer, forward_flatten_layer5) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  FlattenLayer flatten_layer(2, 3); // 1 2
  const auto status = flatten_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_EQ(outputs.size(), 1);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8);
  ASSERT_EQ(shapes.at(2), 24 * 32);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 2);

  ASSERT_EQ(raw_shapes.at(0), 8);
  ASSERT_EQ(raw_shapes.at(1), 24 * 32);

  uint32_t size1 = inputs.front()->size();
  uint32_t size2 = outputs.front()->size();
  ASSERT_EQ(size1, size2);

  for (uint32_t i = 0; i < size1; ++i) {
    ASSERT_EQ(inputs.front()->index(i), outputs.front()->index(i));
  }
}