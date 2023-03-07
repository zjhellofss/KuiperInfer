//
// Created by fss on 22-12-21.
//
#include <gtest/gtest.h>
#include "data/tensor.hpp"

TEST(test_reshape, reshape1) {
  using namespace kuiper_infer;
  Tensor<float> tensor1(3, 224, 224);
  tensor1.Rand();
  Tensor<float> tensor2 = tensor1;
  tensor1.Reshape({224, 224, 3});

  const auto &raw_shapes = tensor1.raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 3);
  ASSERT_EQ(raw_shapes.at(0), 224);
  ASSERT_EQ(raw_shapes.at(1), 224);
  ASSERT_EQ(raw_shapes.at(2), 3);

  ASSERT_EQ(tensor1.size(), tensor2.size());
  uint32_t size = tensor1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor1.index(i), tensor2.index(i));
  }
}

TEST(test_reshape, reshape2) {
  using namespace kuiper_infer;
  Tensor<float> tensor1(3, 224, 224);
  tensor1.Rand();
  Tensor<float> tensor2 = tensor1;
  tensor1.Reshape({672, 224});

  const auto &raw_shapes = tensor1.raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 2);
  ASSERT_EQ(raw_shapes.at(0), 672);
  ASSERT_EQ(raw_shapes.at(1), 224);

  ASSERT_EQ(tensor1.size(), tensor2.size());
  uint32_t size = tensor1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor1.index(i), tensor2.index(i));
  }
}

TEST(test_reshape, reshape3) {
  using namespace kuiper_infer;
  Tensor<float> tensor1(3, 224, 224);
  tensor1.Rand();
  Tensor<float> tensor2 = tensor1;
  tensor1.Reshape({150528});

  const auto &raw_shapes = tensor1.raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 1);
  ASSERT_EQ(raw_shapes.at(0), 150528);

  ASSERT_EQ(tensor1.size(), tensor2.size());
  uint32_t size = tensor1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor1.index(i), tensor2.index(i));
  }
}

