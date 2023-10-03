// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fushenshen on 2023/3/15.
#include "winograd.hpp"
#include <glog/logging.h>
#include "tick.hpp"
namespace kuiper_infer {

static void WinogradTransformG(const arma::fmat& g, arma::fmat& transform_g) {
  arma::fmat Gg(6, 3);
  for (int32_t i = 0; i < 3; ++i) {
    float g0 = g.at(i, 0);
    float g1 = g.at(i, 1);
    float g2 = g.at(i, 2);
    float* Gg_ptr = Gg.colptr(i);

    *(Gg_ptr + 0) = g0 / 4.f;
    *(Gg_ptr + 1) = ((-g0 - g1 - g2) / 2.f) / 3.f;
    *(Gg_ptr + 2) = ((-g0 + g1 - g2) / 2.f) / 3.f;
    *(Gg_ptr + 3) = (g0 / 8.f + g1 / 4.f + g2 / 2.f) / 3.f;
    *(Gg_ptr + 4) = (g0 / 8.f - g1 / 4.f + g2 / 2.f) / 3.f;
    *(Gg_ptr + 5) = g2;
  }

  const arma::fmat& Ggt = Gg.t();
  transform_g = arma::fmat(6, 6);
  for (int32_t i = 0; i < 6; ++i) {
    const float* Ggt_ptr = Ggt.colptr(i);
    const float Gg0 = *(Ggt_ptr + 0);
    const float Gg1 = *(Ggt_ptr + 1);
    const float Gg2 = *(Ggt_ptr + 2);

    transform_g.at(0, i) = Gg0 / 4.f;
    transform_g.at(1, i) = ((-Gg0 - Gg1 - Gg2) / 2.f) / 3.f;
    transform_g.at(2, i) = ((-Gg0 + Gg1 - Gg2) / 2.f) / 3.f;
    transform_g.at(3, i) = (Gg0 / 8.f + Gg1 / 4.f + Gg2 / 2.f) / 3.f;
    transform_g.at(4, i) = (Gg0 / 8.f - Gg1 / 4.f + Gg2 / 2.f) / 3.f;
    transform_g.at(5, i) = Gg2;
  }
}

static void Winograd4x32(const arma::fmat& transform_g, float* in[6], float Y[16]) {
  CHECK_EQ(transform_g.empty(), false);
  CHECK(transform_g.n_cols == 6 && transform_g.n_rows == 6);

  float BTd[6][6];
  for (int32_t i = 0; i < 6; ++i) {
    const float d0 = in[0][i];
    const float d1 = in[1][i];
    const float d2 = in[2][i];
    const float d3 = in[3][i];
    const float d4 = in[4][i];
    const float d5 = in[5][i];

    BTd[0][i] = (d0 * 4) - d2 * 5 + d4;
    BTd[1][i] = -(d1 * 4) - d2 * 4 + d3 + d4;
    BTd[2][i] = (d1 * 4) - d2 * 4 - d3 + d4;
    BTd[3][i] = -(d1 * 2) - d2 + d3 * 2 + d4;
    BTd[4][i] = (d1 * 2) - d2 - d3 * 2 + d4;
    BTd[5][i] = (d1 * 4) - d3 * 5 + d5;
  }

  float V[6][6];
  for (int32_t i = 0; i < 6; ++i) {
    const float BTd0 = BTd[i][0];
    const float BTd1 = BTd[i][1];
    const float BTd2 = BTd[i][2];
    const float BTd3 = BTd[i][3];
    const float BTd4 = BTd[i][4];
    const float BTd5 = BTd[i][5];
    V[i][0] = ((BTd0 * 4) - (BTd2 * 4) - BTd2 + BTd4) * transform_g.at(0, i);
    V[i][1] = (-(BTd1 * 4) - (BTd2 * 4) + BTd3 + BTd4) * transform_g.at(1, i);
    V[i][2] = ((BTd1 * 4) - (BTd2 * 4) - BTd3 + BTd4) * transform_g.at(2, i);
    V[i][3] = (-(BTd1 * 2) - BTd2 + BTd3 * 2 + BTd4) * transform_g.at(3, i);
    V[i][4] = ((BTd1 * 2) - BTd2 - BTd3 * 2 + BTd4) * transform_g.at(4, i);
    V[i][5] = ((BTd1 * 4) - (BTd3 * 4) - BTd3 + BTd5) * transform_g.at(5, i);
  }

  float ATM[4][6];
  for (int32_t i = 0; i < 6; ++i) {
    const float M0 = V[0][i];
    const float M1 = V[1][i];
    const float M2 = V[2][i];
    const float M3 = V[3][i];
    const float M4 = V[4][i];
    const float M5 = V[5][i];

    ATM[0][i] = M0 + M1 + M2 + M3 + M4;
    ATM[1][i] = M1 - M2 + (M3 * 2) - (M4 * 2);
    ATM[2][i] = M1 + M2 + (M3 * 4) + (M4 * 4);
    ATM[3][i] = M1 - M2 + (M3 * 8) - (M4 * 8) + M5;
  }

  for (int32_t i = 0; i < 4; ++i) {
    const float ATM0 = ATM[i][0];
    const float ATM1 = ATM[i][1];
    const float ATM2 = ATM[i][2];
    const float ATM3 = ATM[i][3];
    const float ATM4 = ATM[i][4];
    const float ATM5 = ATM[i][5];

    Y[i * 4 + 0] = float(ATM0 + ATM1 + ATM2 + ATM3 + ATM4);
    Y[i * 4 + 1] = float(ATM1 - ATM2 + (ATM3 * 2) - (ATM4 * 2));
    Y[i * 4 + 2] = float(ATM1 + ATM2 + (ATM3 * 4) + (ATM4 * 4));
    Y[i * 4 + 3] = float(ATM1 - ATM2 + (ATM3 * 8) - (ATM4 * 8) + ATM5);
  }
}

void Convolution3x3s1(const std::shared_ptr<Tensor<float>>& input,
                      std::shared_ptr<Tensor<float>>& output,
                      const std::vector<sftensor>& weights) {
  CHECK(!input->empty()) << "Input for winograd is empty!";

  const uint32_t input_h = input->rows();
  const uint32_t input_w = input->cols();
  const uint32_t input_channels = input->channels();

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t kernel_count = weights.size();

  const uint32_t output_h = std::floor((int32_t(input_h) - int32_t(kernel_h)) / stride_h + 1);
  const uint32_t output_w = std::floor((int32_t(input_w) - int32_t(kernel_w)) / stride_w + 1);
  if (output == nullptr || output->empty()) {
    output = std::make_shared<ftensor>(kernel_count, output_h, output_w);
  }
  CHECK(output != nullptr && !output->empty());
  CHECK(output->rows() == output_h && output->cols() == output_w);

  const uint32_t tile_num_w =
      (int32_t(input_w) - 6) / 4 + ((int32_t(input_w) - 6) % 4 > 0 ? 1 : 0) + 1;

  const uint32_t tile_num_h =
      (int32_t(input_h) - 6) / 4 + ((int32_t(input_h) - 6) % 4 > 0 ? 1 : 0) + 1;

  const uint32_t padded_in_w = 4 * tile_num_w + 2;
  const uint32_t padded_in_h = 4 * tile_num_h + 2;

  sftensor padded_input;
  if (padded_in_w != input_w && padded_in_h != input_h) {
    padded_input = std::make_shared<ftensor>(input_channels, padded_in_h, padded_in_w);
    for (uint32_t c = 0; c < input_channels; ++c) {
      arma::fmat& padded_input_channel = padded_input->slice(c);
      const arma::fmat& input_channel = input->slice(c);
      for (uint32_t w = 0; w < input_w; ++w) {
        float* padded_col_ptr = padded_input_channel.colptr(w);
        const float* col_ptr = input_channel.colptr(w);
        for (uint32_t h = 0; h < input_h; ++h) {
          *(padded_col_ptr + h) = *(col_ptr + h);
        }
      }
    }
  } else {
    padded_input = input;
  }

#pragma omp parallel for
  for (uint32_t k = 0; k < kernel_count; ++k) {
    for (uint32_t input_c = 0; input_c < input_channels; ++input_c) {
      const auto& weight = weights.at(k);
      arma::fmat transform_g;
      WinogradTransformG(weight->slice(input_c), transform_g);
      const arma::fmat& padded_input_channel = padded_input->slice(input_c);

#pragma omp parallel for
      for (uint32_t tile_ind_x = 0; tile_ind_x < tile_num_w; ++tile_ind_x) {
        for (uint32_t tile_ind_y = 0; tile_ind_y < tile_num_h; ++tile_ind_y) {
          uint32_t tile_x = tile_ind_x * 4;
          uint32_t tile_y = tile_ind_y * 4;
          float* in[6];
          for (int32_t i = 0; i < 6; i++) {
            float* padded_input_channel_colptr =
                (float*)padded_input_channel.colptr(tile_x + i) + tile_y;
            in[i] = padded_input_channel_colptr;
          }

          float Y[16];
          Winograd4x32(transform_g, in, Y);
          const arma::fmat& out_channel = output->slice(k);
#pragma omp simd
          for (int32_t i = 0; i < 4; ++i) {
            if (tile_x + i < out_channel.n_cols) {
              float* col_ptr = (float*)out_channel.colptr(tile_x + i);
              for (int32_t j = 0; j < 4; ++j) {
                if (tile_y + j < out_channel.n_rows) {
                  *(col_ptr + tile_y + j) += Y[i * 4 + j];
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace kuiper_infer