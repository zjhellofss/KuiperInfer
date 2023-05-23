//
// Created by fss on 23-5-23.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <vector>
#include "data/tensor.hpp"
TEST(test_multi_tensor, tensor_create) {
  using namespace kuiper_infer;
  ftensor_nd<3, 4> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 2);
  CHECK_EQ(shapes.at(0), 3);
  CHECK_EQ(shapes.at(1), 4);
}

TEST(test_multi_tensor, size) {
  using namespace kuiper_infer;
  ftensor_nd<3, 4, 5> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 60);
}

TEST(test_multi_tensor, raw_ptr1) {
  using namespace kuiper_infer;
  ftensor_nd<3, 4, 5> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 3);
  CHECK_EQ(shapes.at(0), 3);
  CHECK_EQ(shapes.at(1), 4);
  CHECK_EQ(shapes.at(2), 5);
  float* ptr1 = tensor.raw_ptr();
  CHECK(ptr1 != nullptr);
  ptr1 += 1;
  float* ptr2 = tensor.raw_ptr();
  CHECK_EQ(ptr1, ptr2 + 1);
}

TEST(test_multi_tensor, fill1) {
  using namespace kuiper_infer;
  ftensor_nd<2, 2, 3> tensor;
  std::vector<float> values;
  for (int i = 1; i <= 12; ++i) {
    values.push_back(float(i));
  }
  tensor.Fill(values);

  float* ptr = tensor.raw_ptr(0);
  for (uint32_t i = 1; i <= 12; ++i) {
    ASSERT_EQ(*(ptr + i), i);
  }
}

TEST(test_multi_tensor, fill2) {
  using namespace kuiper_infer;
  ftensor_nd<2, 2, 3> tensor;
  tensor.Fill(3.f);

  float* ptr = tensor.raw_ptr(0);
  for (uint32_t i = 1; i <= 12; ++i) {
    ASSERT_EQ(*(ptr + i), 3.f);
  }
}

TEST(test_multi_tensor, raw_ptr2) {
  using namespace kuiper_infer;
  ftensor_nd<2, 2, 3> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 3);
  CHECK_EQ(shapes.at(0), 2);
  CHECK_EQ(shapes.at(1), 2);
  CHECK_EQ(shapes.at(2), 3);

  std::vector<float> values;
  for (int i = 1; i <= 12; ++i) {
    values.push_back(float(i));
  }
  tensor.Fill(values);

  float* ptr = tensor.raw_ptr(std::vector<uint32_t>{1});
  for (uint32_t i = 0; i < 6; ++i) {
    ASSERT_EQ(*(ptr + i), i + 2);
  }
}

TEST(test_multi_tensor, raw_ptr3) {
  using namespace kuiper_infer;
  ftensor_nd<2, 2, 3> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 3);
  CHECK_EQ(shapes.at(0), 2);
  CHECK_EQ(shapes.at(1), 2);
  CHECK_EQ(shapes.at(2), 3);

  std::vector<float> values;
  for (int i = 1; i <= 12; ++i) {
    values.push_back(float(i));
  }
  tensor.Fill(values);

  float* ptr = tensor.raw_ptr(std::vector<uint32_t>{1, 0, 2});
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*(ptr + i), 9 + i);
  }
}

TEST(test_multi_tensor, raw_ptr4) {
  using namespace kuiper_infer;
  ftensor_nd<2, 2, 3> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 3);
  CHECK_EQ(shapes.at(0), 2);
  CHECK_EQ(shapes.at(1), 2);
  CHECK_EQ(shapes.at(2), 3);

  std::vector<float> values;
  for (int i = 1; i <= 12; ++i) {
    values.push_back(float(i));
  }
  tensor.Fill(values);

  float* ptr = tensor.raw_ptr(std::vector<uint32_t>{0, 0, 1});
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*(ptr + i), 2 + i);
  }
}

TEST(test_multi_tensor, raw_ptr5) {
  using namespace kuiper_infer;
  ftensor_nd<2, 2, 3> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 3);
  CHECK_EQ(shapes.at(0), 2);
  CHECK_EQ(shapes.at(1), 2);
  CHECK_EQ(shapes.at(2), 3);

  std::vector<float> values;
  for (int i = 1; i <= 12; ++i) {
    values.push_back(float(i));
  }
  tensor.Fill(values);

  float* ptr = tensor.raw_ptr(std::vector<uint32_t>{0, 1, 2});
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*(ptr + i), 6 + i);
  }
}

TEST(test_multi_tensor, raw_ptr6) {
  using namespace kuiper_infer;
  ftensor_nd<4, 3> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 2);
  CHECK_EQ(shapes.at(0), 4);
  CHECK_EQ(shapes.at(1), 3);

  std::vector<float> values;
  for (int i = 1; i <= 12; ++i) {
    values.push_back(float(i));
  }
  tensor.Fill(values);

  float* ptr = tensor.raw_ptr(std::vector<uint32_t>{1, 0});
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*(ptr + i), 4 + i);
  }
}

TEST(test_multi_tensor, raw_ptr7) {
  using namespace kuiper_infer;
  ftensor_nd<4, 3> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 2);
  CHECK_EQ(shapes.at(0), 4);
  CHECK_EQ(shapes.at(1), 3);

  std::vector<float> values;
  for (int i = 1; i <= 12; ++i) {
    values.push_back(float(i));
  }
  tensor.Fill(values);

  float* ptr = tensor.raw_ptr(std::vector<uint32_t>{2, 2});
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*(ptr + i), 10 + i);
  }
}

TEST(test_multi_tensor, raw_ptr8) {
  using namespace kuiper_infer;
  ftensor_nd<3, 3, 3> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 3);
  CHECK_EQ(shapes.at(0), 3);
  CHECK_EQ(shapes.at(1), 3);
  CHECK_EQ(shapes.at(2), 3);

  std::vector<float> values;
  for (int i = 1; i <= 27; ++i) {
    values.push_back(float(i));
  }
  tensor.Fill(values);

  float* ptr = tensor.raw_ptr(std::vector<uint32_t>{1, 1, 2});
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*(ptr + i), 15 + i);
  }

  float* ptr1 = tensor.raw_ptr(std::vector<uint32_t>{1, 2, 2});
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*(ptr1 + i), 18 + i);
  }

  float* ptr2 = tensor.raw_ptr(std::vector<uint32_t>{2, 0, 2});
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*(ptr2 + i), 21 + i);
  }
}

TEST(test_multi_tensor, raw_ptr9) {
  using namespace kuiper_infer;
  ftensor_nd<3, 3, 3> tensor;
  const auto& shapes = tensor.raw_shapes();
  CHECK(shapes.size() == 3);
  CHECK_EQ(shapes.at(0), 3);
  CHECK_EQ(shapes.at(1), 3);
  CHECK_EQ(shapes.at(2), 3);

  std::vector<float> values;
  for (int i = 1; i <= 27; ++i) {
    values.push_back(float(i));
  }
  tensor.Fill(values);

  float* ptr = tensor.raw_ptr(std::vector<uint32_t>{1});
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*(ptr + i), 10 + i);
  }

  float* ptr1 = tensor.raw_ptr(std::vector<uint32_t>{1, 2});
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(*(ptr1 + i), 16 + i);
  }
}
