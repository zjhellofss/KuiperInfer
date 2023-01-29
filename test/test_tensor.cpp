//
// Created by fss on 23-1-15.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"

TEST(test_tensor, tensor_init1) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 224, 224);
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);
  ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TEST(test_tensor, copy_construct1) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 224, 224);
  Tensor<float> f2(f1);
  ASSERT_EQ(f2.channels(), 3);
  ASSERT_EQ(f2.rows(), 224);
  ASSERT_EQ(f2.cols(), 224);

  ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, copy_construct2) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 2, 1);
  Tensor<float> f2(3, 224, 224);
  f1 = f2;
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, move_construct1) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 2, 1);
  Tensor<float> f2(3, 224, 224);
  f1 = std::move(f2);
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_EQ(f2.data().memptr(), nullptr);
}

TEST(test_tensor, move_construct2) {
  using namespace kuiper_infer;
  Tensor<float> f2(3, 224, 224);
  Tensor<float> f1(std::move(f2));
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_EQ(f2.data().memptr(), nullptr);
}

TEST(test_tensor, tensor_init2) {
  using namespace kuiper_infer;
  Tensor<float> f1(std::vector<uint32_t>{12, 384, 224});
  ASSERT_EQ(f1.channels(), 12);
  ASSERT_EQ(f1.rows(), 384);
  ASSERT_EQ(f1.cols(), 224);
  ASSERT_EQ(f1.size(), 224 * 384 * 12);
}

TEST(test_tensor, set_data) {
  using namespace kuiper_infer;
  Tensor<float> f2(3, 224, 224);
  arma::fcube cube1(224, 224, 3);
  cube1.randn();
  f2.set_data(cube1);

  ASSERT_TRUE(arma::approx_equal(f2.data(), cube1, "absdiff", 1e-4));
}

TEST(test_tensor, data) {
  using namespace kuiper_infer;
  Tensor<float> f2(3, 224, 224);
  f2.Fill(1.f);
  arma::fcube cube1(224, 224, 3);
  cube1.fill(1.);
  f2.set_data(cube1);

  ASSERT_TRUE(arma::approx_equal(f2.data(), cube1, "absdiff", 1e-4));
}

TEST(test_tensor, empty) {
  using namespace kuiper_infer;
  Tensor<float> f2;
  ASSERT_EQ(f2.empty(), true);

  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
}

TEST(test_tensor, transform) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  f3.Transform([](const float &value) {
    return 1.f;
  });
  for (int i = 0; i < f3.size(); ++i) {
    ASSERT_EQ(f3.index(i), 1.f);
  }
}

TEST(test_tensor, clone) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  f3.Rand();

  const auto &f4 = f3.Clone();
  assert(f4->data().memptr() != f3.data().memptr());
  ASSERT_EQ(f4->size(), f3.size());
  for (int i = 0; i < f3.size(); ++i) {
    ASSERT_EQ(f3.index(i), f4->index(i));
  }
}

TEST(test_tensor, raw_ptr) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.raw_ptr(), f3.data().mem);
}

TEST(test_tensor, index1) {
  using namespace kuiper_infer;
  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(1);
  }
  f3.Fill(values);
  for (int i = 0; i < 27; ++i) {
    ASSERT_EQ(f3.index(i), 1);
  }
}

TEST(test_tensor, index2) {
  using namespace kuiper_infer;
  Tensor<float> f3(3, 3, 3);
  f3.index(3) = 4;
  ASSERT_EQ(f3.index(3), 4);
}

TEST(test_tensor, flatten) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.Fill(values);
  f3.Flatten();
  ASSERT_EQ(f3.channels(), 1);
  ASSERT_EQ(f3.rows(), 27);
  ASSERT_EQ(f3.cols(), 1);
}

TEST(test_tensor, fill1) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.Fill(values);
  int index = 0;
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < f3.rows(); ++i) {
      for (int j = 0; j < f3.cols(); ++j) {
        ASSERT_EQ(f3.at(c, i, j), index);
        index += 1;
      }
    }
  }
}

TEST(test_tensor, create) {
  using namespace kuiper_infer;
  const std::shared_ptr<ftensor> &tensor_ptr = Tensor<float>::Create(3, 32, 32);
  ASSERT_EQ(tensor_ptr->empty(), false);
  ASSERT_EQ(tensor_ptr->channels(), 3);
  ASSERT_EQ(tensor_ptr->rows(), 32);
  ASSERT_EQ(tensor_ptr->cols(), 32);
}

TEST(test_tensor, fill2) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  std::vector<float> values;

  f3.Fill(1.f);
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < f3.rows(); ++i) {
      for (int j = 0; j < f3.cols(); ++j) {
        ASSERT_EQ(f3.at(c, i, j), 1.f);
      }
    }
  }
}

TEST(test_tensor, add1) {
  using namespace kuiper_infer;
  const auto &f1 = std::make_shared<Tensor<float>>(3, 224, 224);
  f1->Fill(1.f);
  const auto &f2 = std::make_shared<Tensor<float>>(3, 224, 224);
  f2->Fill(2.f);
  const auto &f3 = Tensor<float>::ElementAdd(f2, f1);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 3.f);
  }
}

TEST(test_tensor, mul1) {
  using namespace kuiper_infer;
  const auto &f1 = std::make_shared<Tensor<float>>(3, 224, 224);
  f1->Fill(3.f);
  const auto &f2 = std::make_shared<Tensor<float>>(3, 224, 224);
  f2->Fill(2.f);
  const auto &f3 = Tensor<float>::ElementMultiply(f2, f1);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 6.f);
  }
}

TEST(test_tensor, shapes) {
  using namespace kuiper_infer;
  Tensor<float> f3(2, 3, 4);
  const std::vector<uint32_t> shapes = f3.shapes();
  ASSERT_EQ(shapes.at(0), 2);
  ASSERT_EQ(shapes.at(1), 3);
  ASSERT_EQ(shapes.at(2), 4);
}

TEST(test_tensor, raw_shapes1) {
  using namespace kuiper_infer;
  Tensor<float> f3(2, 3, 4);
  f3.ReRawshape({24});
  const auto &shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 1);
  ASSERT_EQ(shapes.at(0), 24);
}

TEST(test_tensor, raw_shapes2) {
  using namespace kuiper_infer;
  Tensor<float> f3(2, 3, 4);
  f3.ReRawshape({4, 6});
  const auto &shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 2);
  ASSERT_EQ(shapes.at(0), 4);
  ASSERT_EQ(shapes.at(1), 6);
}

TEST(test_tensor, raw_shapes3) {
  using namespace kuiper_infer;
  Tensor<float> f3(2, 3, 4);
  f3.ReRawshape({4, 3, 2});
  const auto &shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 3);
  ASSERT_EQ(shapes.at(0), 4);
  ASSERT_EQ(shapes.at(1), 3);
  ASSERT_EQ(shapes.at(2), 2);
}

TEST(test_tensor, raw_view1) {
  using namespace kuiper_infer;
  Tensor<float> f3(2, 3, 4);
  f3.ReRawView({24});
  const auto &shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 1);
  ASSERT_EQ(shapes.at(0), 24);
}

TEST(test_tensor, raw_view2) {
  using namespace kuiper_infer;
  Tensor<float> f3(2, 3, 4);
  f3.ReRawView({4, 6});
  const auto &shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 2);
  ASSERT_EQ(shapes.at(0), 4);
  ASSERT_EQ(shapes.at(1), 6);
}

TEST(test_tensor, raw_view3) {
  using namespace kuiper_infer;
  Tensor<float> f3(2, 3, 4);
  f3.ReRawView({4, 3, 2});
  const auto &shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 3);
  ASSERT_EQ(shapes.at(0), 4);
  ASSERT_EQ(shapes.at(1), 3);
  ASSERT_EQ(shapes.at(2), 2);
}

TEST(test_tensor, padding1) {
  using namespace kuiper_infer;
  Tensor<float> tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  tensor.Padding({1, 2, 3, 4}, 0);
  ASSERT_EQ(tensor.rows(), 7);
  ASSERT_EQ(tensor.cols(), 12);

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        if (c_ == 0 || r == 0) {
          ASSERT_EQ(tensor.at(c, r, c_), 0);
        }
        index += 1;
      }
    }
  }
}

TEST(test_tensor, padding2) {
  using namespace kuiper_infer;
  ftensor tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  tensor.Padding({2, 2, 2, 2}, 3.14f);
  ASSERT_EQ(tensor.rows(), 8);
  ASSERT_EQ(tensor.cols(), 9);

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        if (c_ <= 1 || r <= 1) {
          ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
        } else if (c >= tensor.cols() - 1 || r >= tensor.rows() - 1) {
          ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
        }
        if ((r >= 2 && r <= 5) && (c_ >= 2 && c_ <= 6)) {
          ASSERT_EQ(tensor.at(c, r, c_), 1.f);
        }
        index += 1;
      }
    }
  }
}

TEST(test_tensor, review) {
  using namespace kuiper_infer;
  Tensor<float> tensor(3, 4, 5);
  std::vector<float> values;
  for (int i = 0; i < 60; ++i) {
    values.push_back(float(i));
  }

  tensor.Fill(values);

  tensor.ReRawView({4, 3, 5});
  const auto &data = tensor.at(0);
  int index = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      ASSERT_EQ(data.at(i, j), index);
      index += 1;
    }
  }
}

TEST(test_tensor, ones) {
  using namespace kuiper_infer;
  Tensor<float> tensor(3, 4, 5);
  tensor.Ones();
  for (int i = 0; i < tensor.size(); ++i) {
    ASSERT_EQ(tensor.index(i), 1.f);
  }
}

TEST(test_tensor, get_data) {
  using namespace kuiper_infer;
  Tensor<float> tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  arma::fmat in2(4, 5);
  const arma::fmat &in1 = tensor.at(0);
  tensor.at(0) = in2;
  const arma::fmat &in3 = tensor.at(0);
  ASSERT_EQ(in1.memptr(), in3.memptr());
}

TEST(test_tensor, at1) {
  using namespace kuiper_infer;
  Tensor<float> tensor(3, 4, 5);
  tensor.at(0, 1, 2) = 2;
  ASSERT_EQ(tensor.at(0, 1, 2), 2);
}

TEST(test_tensor, at2) {
  using namespace kuiper_infer;
  Tensor<float> tensor(3, 4, 5);
  arma::fmat f(4, 5);
  f.fill(1.f);
  tensor.at(0) = f;
  ASSERT_TRUE(arma::approx_equal(f, tensor.at(0), "absdiff", 1e-4));
}