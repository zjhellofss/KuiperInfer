//
// Created by fss on 23-1-15.
//

#include <gtest/gtest.h>
#include "data/tensor.hpp"
using namespace kuiper_infer;

TEST(test_tensor, tensor_init) {
  Tensor<float> f1(3, 224, 224);
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);
  ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TEST(test_tensor, tensor_rand) {
  Tensor<float> f1(3, 10, 10);
  f1.Rand();
  auto cpu_data = f1.to_cpu_data();
  for (int i = 0; i < f1.size(); i++) {
    std::cout << cpu_data.get()[i] << std::endl;
  }
}
