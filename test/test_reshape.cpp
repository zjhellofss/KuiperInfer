//
// Created by fss on 22-12-21.
//
#include <gtest/gtest.h>
#include "data/tensor.hpp"

TEST(test_reshape, reshape1) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({1, 2, 4});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_reshape, reshape2) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({1, 8});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_reshape, reshape3) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({2, 4});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_reshape, reshape4) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  tensor.ReRawshape({2, 1, 4});

  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}

TEST(test_reshape, reshape5) {
  using namespace kuiper_infer;
  Tensor<float> tensor(8, 24, 32);
  tensor.Rand();

  ASSERT_EQ(tensor.channels(), 8);
  ASSERT_EQ(tensor.rows(), 24);
  ASSERT_EQ(tensor.cols(), 32);

  std::shared_ptr<Tensor<float>> tensor1 = tensor.Clone();
  tensor1->ReRawshape({32, 24, 8});

  ASSERT_EQ(tensor.size(), tensor1->size());
  const uint32_t size = tensor.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor.index(i), tensor1->index(i));
  }
}

TEST(test_reshape, reshape6) {
  using namespace kuiper_infer;
  Tensor<float> tensor(8, 24, 32);
  tensor.Rand();

  ASSERT_EQ(tensor.channels(), 8);
  ASSERT_EQ(tensor.rows(), 24);
  ASSERT_EQ(tensor.cols(), 32);

  std::shared_ptr<Tensor<float>> tensor1 = tensor.Clone();
  tensor1->ReRawshape({32, 24, 8});

  ASSERT_EQ(tensor.size(), tensor1->size());
  const uint32_t size = tensor.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_EQ(tensor.index(i), tensor1->index(i));
  }
}

TEST(test_reshape, reshape7) {
  using namespace kuiper_infer;
  std::vector<float> values{1.f, 3.f, 2.f, 4.f, 5.f, 7.f, 6.f, 8.f};
  Tensor<float> tensor(2, 2, 2);
  tensor.Fill(values);
  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(tensor.index(i), float(i +1));
  }
}
