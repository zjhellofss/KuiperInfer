//
// Created by fss on 23-1-11.
//
#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include "data/tensor.hpp"

static void BM_MatrixMultiplyDynamic(benchmark::State &state) {
  std::vector<arma::fmat> inputs;
  for (int i = 0; i < 128; ++i) {
    arma::fmat input(32, 1280);
    input.randn();
    inputs.push_back(input);
  }
  arma::fmat input2(1280, 2560);
  arma::fmat output_final(32, 2560);
  output_final.randn();
  std::vector<arma::fmat> outputs(128);

  for (auto _ : state) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < 128; ++i) {
      const arma::fmat &output = inputs.at(i) * input2;
      outputs.at(i) = output;
    }
  }

  for (int i = 0; i < outputs.size(); ++i) {
    output_final += outputs.at(i);
  }
  CHECK_NE(output_final.at(0, 0), std::numeric_limits<float>::max() - 1.f);
}

static void BM_MatrixMultiplyStatic(benchmark::State &state) {
  std::vector<arma::fmat> inputs;
  for (int i = 0; i < 128; ++i) {
    arma::fmat input(32, 1280);
    input.randn();
    inputs.push_back(input);
  }
  arma::fmat input2(1280, 2560);
  arma::fmat output_final(32, 2560);
  output_final.randn();
  std::vector<arma::fmat> outputs(128);

#pragma omp parallel for
  for (int i = 0; i < 128; ++i) {
    const arma::fmat &output = inputs.at(i) * input2;
    outputs.at(i) = output;
  }

  for (auto _ : state) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < 128; ++i) {
      const arma::fmat &output = inputs.at(i) * input2;
      outputs.at(i) = output;
    }
  }
  for (int i = 0; i < outputs.size(); ++i) {
    output_final += outputs.at(i);
  }
  CHECK_NE(output_final.at(0, 0), std::numeric_limits<float>::max() - 1.f);
}

BENCHMARK(BM_MatrixMultiplyDynamic);
BENCHMARK(BM_MatrixMultiplyStatic);
