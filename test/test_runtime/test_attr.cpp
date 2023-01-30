//
// Created by fss on 23-1-29.
//
#include "runtime/runtime_attr.hpp"
#include <gtest/gtest.h>

TEST(test_runtime, attr_weight_data1) {
  using namespace kuiper_infer;
  RuntimeAttribute runtime_attr;
  std::vector<char> weight_data;
  for (int i = 0; i < 32; ++i) {
    weight_data.push_back((char) i);
  }
  runtime_attr.weight_data = weight_data;
  const auto &result_weight_data = runtime_attr.weight_data;
  ASSERT_EQ(result_weight_data.size(), 32);

  for (int i = 0; i < 32; ++i) {
    ASSERT_EQ(weight_data.at(i), (char) i);
  }
}

TEST(test_runtime, attr_weight_data2) {
  using namespace kuiper_infer;
  RuntimeAttribute runtime_attr;
  runtime_attr.type = RuntimeDataType::kTypeFloat32;
  std::vector<char> weight_data;
  for (int i = 0; i < 32; ++i) {
    weight_data.push_back(0);
  }
  runtime_attr.weight_data = weight_data;

  const auto &result_weight_data = runtime_attr.get<float>();
  ASSERT_EQ(result_weight_data.size(), 8);

  for (int i = 0; i < 32; ++i) {
    ASSERT_EQ(weight_data.at(i), 0.f);
  }
}

TEST(test_runtime, attr_shape) {
  using namespace kuiper_infer;
  RuntimeAttribute runtime_attr;
  runtime_attr.type = RuntimeDataType::kTypeFloat32;
  runtime_attr.shape = std::vector<int>{3, 32, 32};
  ASSERT_EQ(runtime_attr.shape.at(0), 3);

  ASSERT_EQ(runtime_attr.shape.at(1), 32);
  ASSERT_EQ(runtime_attr.shape.at(2), 32);
}
