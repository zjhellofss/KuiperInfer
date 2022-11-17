//
// Created by fss on 22-11-11.
//
#include <vector>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../source/data/blob.hpp"

TEST(test_blob, size) {
  using namespace kuiper_infer;
  const uint32_t channels = 32;
  const uint32_t rows = 16;
  const uint32_t cols = 16;
  Tensor blob(channels, rows, cols);
  ASSERT_EQ(blob.channels(), channels);
  ASSERT_EQ(blob.rows(), rows);
  ASSERT_EQ(blob.cols(), cols);
  ASSERT_EQ(blob.size(), channels * rows * cols);
}

TEST(test_blob, empty) {
  using namespace kuiper_infer;
  Tensor blob;
  ASSERT_EQ(blob.empty(), true);
}

TEST(test_blob, fill1) {
  using namespace kuiper_infer;
  const uint32_t channels = 64;
  const uint32_t rows = 32;
  const uint32_t cols = 32;
  Tensor blob(channels, rows, cols);
  const double value = 3.;
  blob.Fill(value);
  for (int i = 0; i < channels; ++i) {
    for (int j = 0; j < rows; ++j) {
      for (int k = 0; k < cols; ++k) {
        ASSERT_EQ(blob.at(i, j, k), value);
      }
    }
  }
}

TEST(test_blob, fill2) {
  // 测试flatten和fill
  using namespace kuiper_infer;
  const uint32_t size = 128;
  std::vector<double> datas;
  for (uint32_t i = 0; i < size; ++i) {
    datas.push_back(i);
  }
  const uint32_t channels = 8;
  const uint32_t rows = 4;
  const uint32_t cols = 4;
  Tensor blob(channels, rows, cols);
  blob.Fill(datas);
  blob.Flatten();
  ASSERT_EQ(blob.channels(), 1);
  ASSERT_EQ(blob.rows(), 1);
  ASSERT_EQ(blob.cols(), channels * rows * cols);
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(blob.at(0, 0, i), i);
  }
}

TEST(test_blob, padding) {
  using namespace kuiper_infer;
  const uint32_t channels = 2;
  const uint32_t rows = 2;
  const uint32_t cols = 2;
  const uint32_t size = 8;
  Tensor blob(channels, rows, cols);
  blob.Fill(1.);
  blob.Flatten();
  blob.Padding({0, 0, 1, 1}, 2);
  ASSERT_EQ(blob.at(0, 0, 0), 2);
  ASSERT_EQ(blob.at(0, 0, size + 1), 2);
}

TEST(test_blob, padding2) {
  using namespace kuiper_infer;
  const uint32_t channels = 1;
  const uint32_t rows = 2;
  const uint32_t cols = 2;
  Tensor blob(channels, rows, cols);
  blob.Padding({1, 1, 1, 1}, 2);
  for (int i = 0; i < channels; ++i) {
    for (int j = 1; j < rows + 1; ++j) {
      for (int k = 1; k < cols + 1; ++k) {
        ASSERT_EQ(blob.at(i, j, k), 0.);
      }
    }
  }
}