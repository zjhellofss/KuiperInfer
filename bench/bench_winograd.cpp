//
// Created by fss on 22-12-5.
//
#include <benchmark/benchmark.h>
#include <armadillo>
#include <glog/logging.h>
#include "../source/layer/details/convolution_3x3.hpp"

static void BM_ConvWinograd(benchmark::State &state) {
  float g1[] = {1, 2, 3,
                4, 5, 6,
                7, 8, 9};
  float d1[] = {1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16};

  std::vector<std::vector<arma::fmat>> g_arr_c1;
  std::vector<std::vector<arma::fmat>> d_arr_c1;

  int in_channels = 128;
  int out_channels = 256;

  for (int i = 0; i < 32 * 32; ++i) {
    std::vector<arma::fmat> g_arr;
    std::vector<arma::fmat> d_arr;

    for (int c = 0; c < out_channels; ++c) {
      arma::fmat g(g1, 3, 3);
      arma::fmat d(d1, 4, 4);
      g_arr.push_back(g);
      d_arr.push_back(d);
    }
    g_arr_c1.push_back(g_arr);
    d_arr_c1.push_back(d_arr);
  }
  int batch_size = 8;
  for (auto _ : state) {
    for (int b = 0; b < batch_size; ++b) {
#pragma omp parallel for num_threads(64)
      for (int i = 0; i < 32 * 32; ++i) {
        const std::vector<arma::fmat> &g_tmp_arr = g_arr_c1.at(i);
        const std::vector<arma::fmat> &d_tmp_arr = d_arr_c1.at(i);

        arma::fmat GgGT(4, 4);
        arma::fmat result(1, 4);
#pragma omp parallel for num_threads(32)
        for (int o = 0; o < out_channels; ++o) {
          const arma::fmat &g_tmp = g_tmp_arr.at(o);
          WinogradTransformG(g_tmp, GgGT);
          arma::fmat result_sum(1, 4);
          for (int c = 0; c < in_channels; ++c) {
            const arma::fmat &d_tmp = d_tmp_arr.at(c);
            Winograd(GgGT, d_tmp, result);
            result_sum += result;
          }
        }
      }
    }
  }
}

BENCHMARK(BM_ConvWinograd);
