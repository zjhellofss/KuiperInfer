//
// Created by fss on 22-11-24.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../source/layer/details/flatten.hpp"

TEST(test_flatten, flatten1_small) {
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

TEST(test_flatten, flatten2_small) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_flatten, flatten3_small) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({1, 2, 4});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_flatten, flatten4_small) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({1, 8});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_flatten, flatten5_small) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({2, 4});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_flatten, flatten6_small) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({2, 1, 4});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_flatten, reshape) {
  using namespace kuiper_infer;
  Tensor<float> tensor(8, 24, 32);
  tensor.Rand();

  ASSERT_EQ(tensor.channels(), 8);
  ASSERT_EQ(tensor.rows(), 24);
  ASSERT_EQ(tensor.cols(), 32);

  std::shared_ptr<Tensor<float>> tensor1 = tensor.Clone();

  ASSERT_EQ(tensor.size(), tensor1->size());
  const uint32_t size = tensor.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor.index(i), tensor1->index(i));
  }
}

TEST(test_flatten, reshape2) {
  using namespace kuiper_infer;
  Tensor<float> tensor(8, 24, 32);
  tensor.Rand();

  ASSERT_EQ(tensor.channels(), 8);
  ASSERT_EQ(tensor.rows(), 24);
  ASSERT_EQ(tensor.cols(), 32);

  std::shared_ptr<Tensor<float>> tensor1 = tensor.Clone();

  ASSERT_EQ(tensor.size(), tensor1->size());
  const uint32_t size = tensor.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor.index(i), tensor1->index(i));
  }
}

TEST(test_layer, flatten_layer1) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  FlattenLayer flatten_layer(1, 3);
  flatten_layer.Forward(inputs, outputs);
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

TEST(test_layer, flatten_layer2) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  FlattenLayer flatten_layer(1, -1); // 1 3 -> 0 2
  flatten_layer.Forward(inputs, outputs);
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

TEST(test_layer, flatten_layer3) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  FlattenLayer flatten_layer(1, 2); // 0 1
  flatten_layer.Forward(inputs, outputs);
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

TEST(test_layer, flatten_layer4) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  FlattenLayer flatten_layer(1, -2); // 0 1
  flatten_layer.Forward(inputs, outputs);
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

TEST(test_layer, flatten_layer5) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(8, 24, 32);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  FlattenLayer flatten_layer(2, 3); // 1 2
  flatten_layer.Forward(inputs, outputs);
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