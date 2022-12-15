//
// Created by fss on 22-12-4.
//
#include "convolution_3x3.hpp"
#include <armadillo>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <x86intrin.h>

#include "data/tensor.hpp"
#include "tick.hpp"

arma::fmat WinogradTransformG(const arma::fmat &kernel) {
  CHECK(!kernel.empty() && kernel.n_cols == 3 && kernel.n_rows == 3);

  float G1[12] = {1, 0, 0,
                  0.5, 0.5, 0.5,
                  0.5, -0.5, 0.5,
                  0, 0, 1};

  float GT1[12] = {1, 0.5, 0.5, 0,
                   0, 0.5, -0.5, 0,
                   0, 0.5, 0.5, 1};

  arma::fmat G(G1, 3, 4);
  arma::fmat GT(GT1, 4, 3);
  const arma::fmat &Gg = kernel * G;
  const arma::fmat &kernel_g = GT * Gg;
  return kernel_g;
}

arma::fmat Winograd(const arma::fmat &kernel_g, const arma::fmat &feature) {
  CHECK(!kernel_g.empty() && kernel_g.n_cols == 4 && kernel_g.n_rows == 4);
  CHECK(!feature.empty() && feature.n_cols == 4 && feature.n_rows == 4);

  arma::fmat BTd(16, 1);
  arma::fmat V(4, 4);
  arma::fmat AT_UV(8, 1);

  const float *d_mem = feature.memptr();
  __m128 BTd_03 = _mm_loadu_ps((float *) (d_mem));
  __m128 BTd_47 = _mm_loadu_ps((float *) (d_mem) + 4);
  __m128 BTd_811 = _mm_loadu_ps((float *) (d_mem) + 8);
  __m128 BTd_1215 = _mm_loadu_ps((float *) (d_mem) + 12);

  __m128 BTd_0 = _mm_sub_ps(BTd_03, BTd_811);
  __m128 BTd_1 = _mm_add_ps(BTd_47, BTd_811);
  __m128 BTd_2 = _mm_sub_ps(BTd_811, BTd_47);
  __m128 BTd_3 = _mm_sub_ps(BTd_47, BTd_1215);

  _mm_store_ps(BTd.memptr(), BTd_0);
  _mm_store_ps(BTd.memptr() + 4, BTd_1);
  _mm_store_ps(BTd.memptr() + 8, BTd_2);
  _mm_store_ps(BTd.memptr() + 12, BTd_3);

  float BTd0 = *BTd.memptr();
  float BTd4 = *(BTd.memptr() + 4);
  float BTd1 = *(BTd.memptr() + 1);
  float BTd5 = *(BTd.memptr() + 5);
  float BTd2 = *(BTd.memptr() + 2);
  float BTd6 = *(BTd.memptr() + 6);
  float BTd3 = *(BTd.memptr() + 3);
  float BTd7 = *(BTd.memptr() + 7);
  float BTd8 = *(BTd.memptr() + 8);
  float BTd12 = *(BTd.memptr() + 12);
  float BTd9 = *(BTd.memptr() + 9);
  float BTd13 = *(BTd.memptr() + 13);
  float BTd10 = *(BTd.memptr() + 10);
  float BTd14 = *(BTd.memptr() + 14);
  float BTd11 = *(BTd.memptr() + 11);
  float BTd15 = *(BTd.memptr() + 15);

  float *V_mem = V.memptr();

  *(V_mem + 0) = BTd0 - BTd2;
  *(V_mem + 4) = BTd4 - BTd6;
  *(V_mem + 8) = BTd8 - BTd10;
  *(V_mem + 12) = BTd12 - BTd14;

  *(V_mem + 1) = BTd1 + BTd2;
  *(V_mem + 5) = BTd5 + BTd6;
  *(V_mem + 9) = BTd9 + BTd10;
  *(V_mem + 13) = BTd13 + BTd14;

  *(V_mem + 2) = -BTd1 + BTd2;
  *(V_mem + 6) = -BTd5 + BTd6;
  *(V_mem + 10) = -BTd9 + BTd10;
  *(V_mem + 14) = -BTd13 + BTd14;

  *(V_mem + 3) = BTd1 - BTd3;
  *(V_mem + 7) = BTd5 - BTd7;
  *(V_mem + 11) = BTd9 - BTd11;
  *(V_mem + 15) = BTd13 - BTd15;

  const arma::fmat &UV = kernel_g % V;
  const float *UV_mem = UV.memptr();
  __m128 AT_UV03 = _mm_loadu_ps((float *) (UV_mem));
  __m128 AT_UV47 = _mm_loadu_ps((float *) (UV_mem) + 4);
  __m128 AT_UV811 = _mm_loadu_ps((float *) (UV_mem) + 8);
  __m128 AT_UV1215 = _mm_loadu_ps((float *) (UV_mem) + 12);

  __m128 AT_UV0 = _mm_add_ps(AT_UV03, _mm_add_ps(AT_UV47, AT_UV811));
  __m128 AT_UV1 = _mm_sub_ps(AT_UV47, _mm_add_ps(AT_UV811, AT_UV1215));

  _mm_store_ps(AT_UV.memptr(), AT_UV0);
  _mm_store_ps(AT_UV.memptr() + 4, AT_UV1);

  float AT_UV0_ = *(AT_UV.memptr());
  float AT_UV4 = *(AT_UV.memptr() + 4);

  float AT_UV1_ = *(AT_UV.memptr() + 1);
  float AT_UV5 = *(AT_UV.memptr() + 5);

  float AT_UV2 = *(AT_UV.memptr() + 2);
  float AT_UV6 = *(AT_UV.memptr() + 6);

  float AT_UV3 = *(AT_UV.memptr() + 3);
  float AT_UV7 = *(AT_UV.memptr() + 7);

  arma::fmat result(2, 2);
  result[0] = (AT_UV0_ + AT_UV1_ + AT_UV2);
  result[2] = (AT_UV4 + AT_UV5 + AT_UV6);
  result[1] = (AT_UV1_ - AT_UV2 - AT_UV3);
  result[3] = (AT_UV5 - AT_UV6 - AT_UV7);
  return result;
}

TEST(test_winograd, winograd3x3) {
  using namespace kuiper_infer;

  const uint32_t input_h = 9;
  const uint32_t input_w = 9;

  const uint32_t in_channels = 1;
  const uint32_t out_channels = 4;
  const uint32_t elem_size = 81;
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;

  std::vector<float> values;
  for (int i = 0; i < elem_size; ++i) {
    values.push_back(float(i));
  }

  const uint32_t input_h_padding = (4 - input_h % 4) + input_h;
  const uint32_t input_w_padding = (4 - input_w % 4) + input_w;

  Tensor<float> input(1, input_h, input_w);
  input.Fill(values);
  input.Show();
  input.Padding({0, 4 - input_h % 4, 0, 4 - input_w % 4}, 0.f);
  input.Show();

  const uint32_t output_h = uint32_t(std::floor(input_h - kernel_h + 1));
  const uint32_t output_w = uint32_t(std::floor(input_w - kernel_h + 1));

  const uint32_t output_h_padding = uint32_t(std::floor(input_h_padding - kernel_h + 1));
  const uint32_t output_w_padding = uint32_t(std::floor(input_w_padding - kernel_h + 1));

  std::vector<std::shared_ptr<Tensor<float>>> kernels;
  for (int i = 0; i < out_channels; ++i) {
    std::shared_ptr<Tensor<float>> kernel = std::make_shared<Tensor<float>>(1, kernel_h, kernel_w);
    kernel->Fill(1.f);
    kernels.push_back(kernel);
  }

  std::shared_ptr<Tensor<float>>
      output_channels = std::make_shared<Tensor<float>>(out_channels, output_h_padding, output_w_padding);
  std::shared_ptr<Tensor<float>>
      output_channels_clipping = std::make_shared<Tensor<float>>(out_channels, output_h, output_w);

  TICK(OC)
  for (uint32_t oc = 0; oc < out_channels; ++oc) {
    std::shared_ptr<Tensor<float>> kernel = kernels.at(oc);
    arma::fmat &output_channel = output_channels->at(oc);
    arma::fmat &output_channel_clipping = output_channels_clipping->at(oc);

    for (uint32_t ic = 0; ic < in_channels; ++ic) {
      const arma::fmat &input_channel = input.at(ic);
      const arma::fmat &kernel_channel = kernel->at(ic);
      CHECK(kernel_channel.n_cols == 3 && kernel_channel.n_rows == 3);

      const arma::fmat &kernel_channel_g = WinogradTransformG(kernel_channel);

      const uint32_t h_tiles = input_channel.n_rows;
      const uint32_t w_tiles = input_channel.n_cols;
      CHECK(h_tiles % 4 == 0 && w_tiles % 4 == 0);

      for (uint32_t h_tile = 0; h_tile < h_tiles; h_tile += 2) {
        for (uint32_t w_tile = 0; w_tile < w_tiles; w_tile += 2) {
          if (h_tile + 4 <= h_tiles && w_tile + 4 <= w_tiles) {

            arma::fmat tile_mat(4, 4);
            for (uint32_t i = 0; i < 4; ++i) {
              const float *col_ptr = input_channel.colptr(i + w_tile) + h_tile;
              memcpy(tile_mat.memptr() + i * 4, col_ptr, 4 * sizeof(float));
            }
            const arma::fmat &output_tile = Winograd(kernel_channel_g, tile_mat);
            if (w_tile + 2 <= output_w_padding && h_tile + 2 <= output_h_padding) {
              float *output_channel_ptr = output_channel.colptr(w_tile) + h_tile;
              *(output_channel_ptr) += *(output_tile.memptr());
              *(output_channel_ptr + 1) += *(output_tile.memptr() + 1);

              output_channel_ptr = output_channel.colptr(w_tile + 1) + h_tile;
              *(output_channel_ptr) += *(output_tile.memptr() + 2);
              *(output_channel_ptr + 1) += *(output_tile.memptr() + 3);
            }
          }
        }
      }
    }
    output_channel_clipping = output_channel.submat(0, 0, output_h - 1, output_w - 1);
  }
  TOCK(OC)
}
