//
// Created by fss on 22-12-5.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <armadillo>
#include "data/fast_copy.hpp"

TEST(test_copy, copy1024) {
  using namespace kuiper_infer;
  const int size = 1024;
  std::vector<float> in(size);
  for (int i = 0; i < size; ++i) {
    in.at(i) = float(i);
  }

  std::vector<float> out(size);
  FastMemoryCopy(out.data(), in.data(), size);

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(out.at(i), float (i));
  }
}

TEST(test_copy, copy2048) {
  using namespace kuiper_infer;
  const int size = 2048;
  std::vector<float> in(size);
  for (int i = 0; i < size; ++i) {
    in.at(i) = float(i) + size;
  }

  std::vector<float> out(size);
  FastMemoryCopy(out.data(), in.data(), size);

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(out.at(i), float (i)+size);
  }
}


