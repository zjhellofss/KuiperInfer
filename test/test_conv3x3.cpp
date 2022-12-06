//
// Created by fss on 22-12-4.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <armadillo>
#include "tick.hpp"
#include "../source/layer/details/convolution_3x3.hpp"

TEST(test_layer, forward_conv3x3) {
  float g1[] = {1, 2, 3,
                4, 5, 6,
                7, 8, 9};
  float d1[] = {1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16};

  std::vector<std::vector<arma::fmat>> g_arr_c1;
  std::vector<std::vector<arma::fmat>> d_arr_c1;

  int channels = 128;
  for (int i = 0; i < 128 * 128; ++i) {
    std::vector<arma::fmat> g_arr;
    std::vector<arma::fmat> d_arr;

    for (int c = 0; c < channels; ++c) {
      arma::fmat g(g1, 3, 3);
      arma::fmat d(d1, 4, 4);
      g_arr.push_back(g);
      d_arr.push_back(d);
    }
    g_arr_c1.push_back(g_arr);
    d_arr_c1.push_back(d_arr);
  }

  for (int i = 0; i < 128 * 128; ++i) {
    const std::vector<arma::fmat> &g_tmp_arr = g_arr_c1.at(i);
    const std::vector<arma::fmat> &d_tmp_arr = d_arr_c1.at(i);

    for (int c = 0; c < channels; ++c) {
      const arma::fmat &g_tmp = g_tmp_arr.at(c);
      const arma::fmat &d_tmp = d_tmp_arr.at(c);
      arma::fmat GgGT(4, 4);
      arma::fmat result(1, 4);
      WinogradTransformG(g_tmp, GgGT);
      Winograd(GgGT, d_tmp, result);
      ASSERT_EQ(result[0], 348);
      ASSERT_EQ(result[1], 393);
      ASSERT_EQ(result[2], 528);
      ASSERT_EQ(result[3], 573);
    }
  }
}