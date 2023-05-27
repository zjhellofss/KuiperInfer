//
// Created by fss on 23-1-15.
//

#include <gtest/gtest.h>
#include "data/tensor.hpp"

TEST(test_tensor, tensor_init1) {
  using namespace kuiper_infer;

  Tensor<float> f1(3, 224, 224);
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);
  ASSERT_EQ(f1.size(), 224 * 224 * 3);
}
