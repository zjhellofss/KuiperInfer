//
// Created by fss on 23-1-11.
//
#include <benchmark/benchmark.h>
#include "data/tensor.hpp"

static void BM_View1(benchmark::State &state) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(16, 320, 320);
  input->Rand();

  for (auto _ : state) {
    const std::vector<uint32_t> shapes{1, 320, 320 * 16};
    const uint32_t target_channels = shapes.at(0);
    const uint32_t target_rows = shapes.at(1);
    const uint32_t target_cols = shapes.at(2);
    arma::fcube new_data(target_rows, target_cols, target_channels);

    const uint32_t plane_size = target_rows * target_cols;
    for (uint32_t c = 0; c < input->channels(); ++c) {
      for (uint32_t r = 0; r < input->rows(); ++r) {
        for (uint32_t c_ = 0; c_ < input->cols(); ++c_) {
          const uint32_t pos_index = c * input->rows() * input->cols() + r * input->cols() + c_;
          const uint32_t ch = pos_index / plane_size;
          const uint32_t row = (pos_index - ch * plane_size) / target_cols;
          const uint32_t col = (pos_index - ch * plane_size - row * target_cols);
          new_data.at(row, col, ch) = input->at(c, r, c_);
        }
      }
    }
  }
}

static void BM_View2(benchmark::State &state) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(16, 320, 320);
  input->Rand();

  for (auto _ : state) {
    const std::vector<uint32_t> shapes{1, 320, 320 * 16};
    const uint32_t target_channels = shapes.at(0);
    const uint32_t target_rows = shapes.at(1);
    const uint32_t target_cols = shapes.at(2);
    arma::fcube new_data(target_rows, target_cols, target_channels);

    const uint32_t plane_size = target_rows * target_cols;
    for (uint32_t c = 0; c < input->channels(); ++c) {
      const arma::fmat &channel = input->at(c);

      for (uint32_t c_ = 0; c_ < input->cols(); ++c_) {
        const float *colptr = channel.colptr(c_);
        for (uint32_t r = 0; r < input->rows(); ++r) {
          const uint32_t pos_index = c * input->rows() * input->cols() + r * input->cols() + c_;
          const uint32_t ch = pos_index / plane_size;
          const uint32_t row = (pos_index - ch * plane_size) / target_cols;
          const uint32_t col = (pos_index - ch * plane_size - row * target_cols);
          new_data.at(row, col, ch) = *(colptr + r);
        }
      }
    }
  }
}

BENCHMARK(BM_View1);
BENCHMARK(BM_View2);
