//
// Created by fss on 23-1-23.
//
#include <benchmark/benchmark.h>
#include <armadillo>
#include <glog/logging.h>

static void BM_QuantInt(benchmark::State &state) {
  arma::imat i1(4096, 1024);
  arma::imat i2(1024, 512);
  for (auto _ : state) {
    const arma::imat &i3 = (i1 * i2);
    CHECK(i3.n_rows == 4096 && i3.n_cols == 512);
  }
}

static void BM_QuantFloat(benchmark::State &state) {
  arma::fmat f1(4096, 1024);
  arma::fmat f2(1024, 512);
  for (auto _ : state) {
    const arma::fmat &f3 = (f1 * f2);
    CHECK(f3.n_rows == 4096 && f3.n_cols == 512);
  }
}


BENCHMARK(BM_QuantInt);
BENCHMARK(BM_QuantFloat);