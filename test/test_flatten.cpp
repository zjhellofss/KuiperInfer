//
// Created by fss on 22-11-24.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"

TEST(test_layer, test_flatten1_small) {
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

TEST(test_layer, test_flatten2_small) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_layer, test_flatten3_small) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({1, 2, 4});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_layer, test_flatten4_small) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({1, 8});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_layer, test_flatten5_small) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({2, 4});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_layer, test_flatten6_small) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({2, 1, 4});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}