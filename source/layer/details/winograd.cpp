//
// Created by fushenshen on 2023/3/15.
//
#include "winograd.hpp"
#include <glog/logging.h>
#include "tick.hpp"
namespace kuiper_infer {

static arma::fmat WinogradTransformG(const arma::fmat& weight) {
  arma::fmat Gg(6, 3);
  for (int i = 0; i < 3; ++i) {
    float g0 = weight.at(i, 0);
    float g1 = weight.at(i, 1);
    float g2 = weight.at(i, 2);
    float* Gg_colptr = Gg.colptr(i);

    *(Gg_colptr + 0) = g0 / 4.f;
    *(Gg_colptr + 1) = ((-g0 - g1 - g2) / 2.f) / 3.f;
    *(Gg_colptr + 2) = ((-g0 + g1 - g2) / 2.f) / 3.f;
    *(Gg_colptr + 3) = (g0 / 8.f + g1 / 4.f + g2 / 2.f) / 3.f;
    *(Gg_colptr + 4) = (g0 / 8.f - g1 / 4.f + g2 / 2.f) / 3.f;
    *(Gg_colptr + 5) = g2;
  }
  return Gg;
}

static arma::fmat Winograd4x3(const arma::fmat& Gg, float d[6][6]) {
  CHECK_EQ(Gg.empty(), false);
  CHECK_EQ(Gg.n_rows, 6);
  CHECK_EQ(Gg.n_cols, 3);

  arma::fmat BTd(6, 6);
  for (int i = 0; i < 6; ++i) {
    const float d0 = d[0][i];
    const float d1 = d[1][i];
    const float d2 = d[2][i];
    const float d3 = d[3][i];
    const float d4 = d[4][i];
    const float d5 = d[5][i];
    float* BTd_colptr = BTd.colptr(i);

    *(BTd_colptr + 0) = (d0 * 4) - d2 * 5 + d4;
    *(BTd_colptr + 1) = -(d1 * 4) - d2 * 4 + d3 + d4;
    *(BTd_colptr + 2) = (d1 * 4) - d2 * 4 - d3 + d4;
    *(BTd_colptr + 3) = -(d1 * 2) - d2 + d3 * 2 + d4;
    *(BTd_colptr + 4) = (d1 * 2) - d2 - d3 * 2 + d4;
    *(BTd_colptr + 5) = (d1 * 4) - d3 * 5 + d5;
  }

  arma::fmat V(6, 6);
  for (int i = 0; i < 6; ++i) {
    const float BTd0 = BTd.at(i, 0);
    const float BTd1 = BTd.at(i, 1);
    const float BTd2 = BTd.at(i, 2);
    const float BTd3 = BTd.at(i, 3);
    const float BTd4 = BTd.at(i, 4);
    const float BTd5 = BTd.at(i, 5);

    V.at(i, 0) = (BTd0 * 4) - (BTd2 * 4) - BTd2 + BTd4;
    V.at(i, 1) = -(BTd1 * 4) - (BTd2 * 4) + BTd3 + BTd4;
    V.at(i, 2) = (BTd1 * 4) - (BTd2 * 4) - BTd3 + BTd4;
    V.at(i, 3) = -(BTd1 * 2) - BTd2 + BTd3 * 2 + BTd4;
    V.at(i, 4) = (BTd1 * 2) - BTd2 - BTd3 * 2 + BTd4;
    V.at(i, 5) = (BTd1 * 4) - (BTd3 * 4) - BTd3 + BTd5;
  }

  arma::fmat U(6, 6);
  for (int i = 0; i < 6; ++i) {
    const float Gg0 = Gg.at(i, 0);
    const float Gg1 = Gg.at(i, 1);
    const float Gg2 = Gg.at(i, 2);

    U.at(i, 0) = Gg0 / 4.f;
    U.at(i, 1) = ((-Gg0 - Gg1 - Gg2) / 2.f) / 3.f;
    U.at(i, 2) = ((-Gg0 + Gg1 - Gg2) / 2.f) / 3.f;
    U.at(i, 3) = (Gg0 / 8.f + Gg1 / 4.f + Gg2 / 2.f) / 3.f;
    U.at(i, 4) = (Gg0 / 8.f - Gg1 / 4.f + Gg2 / 2.f) / 3.f;
    U.at(i, 5) = Gg2;
  }

  const arma::fmat& M = U % V;
  // calculate matrix A'M;
  arma::fmat ATM(4, 6);
  for (int i = 0; i < 6; ++i) {
    const float* M_colptr = M.colptr(i);
    float M0 = *(M_colptr + 0);
    float M1 = *(M_colptr + 1);
    float M2 = *(M_colptr + 2);
    float M3 = *(M_colptr + 3);
    float M4 = *(M_colptr + 4);
    float M5 = *(M_colptr + 5);

    float* ATM_colptr = ATM.colptr(i);

    *(ATM_colptr + 0) = M0 + M1 + M2 + M3 + M4;
    *(ATM_colptr + 1) = M1 - M2 + (M3 * 2) - (M4 * 2);
    *(ATM_colptr + 2) = M1 + M2 + (M3 * 4) + (M4 * 4);
    *(ATM_colptr + 3) = M1 - M2 + (M3 * 8) - (M4 * 8) + M5;
  }

  arma::fmat Y(4, 4);
  for (int i = 0; i < 4; ++i) {
    const float ATM0 = *(ATM.colptr(0) + i);
    const float ATM1 = *(ATM.colptr(1) + i);
    const float ATM2 = *(ATM.colptr(2) + i);
    const float ATM3 = *(ATM.colptr(3) + i);
    const float ATM4 = *(ATM.colptr(4) + i);
    const float ATM5 = *(ATM.colptr(5) + i);

    float* Y_colptr = Y.colptr(i);
    *(Y_colptr + 0) = float(ATM0 + ATM1 + ATM2 + ATM3 + ATM4);
    *(Y_colptr + 1) = float(ATM1 - ATM2 + (ATM3 * 2) - (ATM4 * 2));
    *(Y_colptr + 2) = float(ATM1 + ATM2 + (ATM3 * 4) + (ATM4 * 4));
    *(Y_colptr + 3) = float(ATM1 - ATM2 + (ATM3 * 8) - (ATM4 * 8) + ATM5);
  }
  return Y;
}

void Convolution3x3s1(const std::shared_ptr<Tensor<float>>& input,
                      std::shared_ptr<Tensor<float>>& output,
                      const sftensor& weights) {
  CHECK(!input->empty()) << "Input for winograd is empty!";
  const uint32_t kernel_h = weights->rows();
  const uint32_t kernel_w = weights->cols();
  const uint32_t input_h = input->rows();
  const uint32_t input_w = input->cols();
  const uint32_t input_channels = input->channels();

  CHECK_EQ(kernel_h, 3);
  CHECK_EQ(kernel_w, 3);
  CHECK_EQ(weights->channels(), input_channels);

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;

  const uint32_t output_h = (input_h - ((kernel_h - 1) + 1)) / stride_h + 1;
  const uint32_t output_w = (input_w - ((kernel_w - 1) + 1)) / stride_w + 1;
  if (output == nullptr || output->empty()) {
    output = std::make_shared<ftensor>(1, output_h, output_w);
  }
  CHECK(output != nullptr && !output->empty());
  CHECK(output->rows() == output_h && output->cols() == output_w);

  const uint32_t tile_num_w =
      (int(input_w) - 6) / 4 + ((int(input_w) - 6) % 4 > 0 ? 1 : 0) + 1;

  const uint32_t tile_num_h =
      (int(input_h) - 6) / 4 + ((int(input_h) - 6) % 4 > 0 ? 1 : 0) + 1;

  const uint32_t padded_in_w = 4 * tile_num_w + 2;
  const uint32_t padded_in_h = 4 * tile_num_h + 2;
  const uint32_t padded_out_w = 4 * tile_num_w;
  const uint32_t padded_out_h = 4 * tile_num_h;

  sftensor padded_input =
      std::make_shared<ftensor>(input_channels, padded_in_h, padded_in_w);
  //
  sftensor padded_out =
      std::make_shared<ftensor>(1, padded_out_h, padded_out_w);
  //
  //  // 填充数据
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

  std::vector<arma::fmat> channels_transform(input_channels);
  for (uint32_t c = 0; c < input_channels; ++c) {
    const arma::fmat& transform_g = WinogradTransformG(weights->slice(c));
    channels_transform.at(c) = transform_g;
  }

#pragma omp parallel for num_threads(8) collapse(2)
  for (uint32_t tile_ind_x = 0; tile_ind_x < tile_num_w; ++tile_ind_x) {
    for (uint32_t tile_ind_y = 0; tile_ind_y < tile_num_h; ++tile_ind_y) {
      for (uint32_t input_c = 0; input_c < input_channels; ++input_c) {
        const arma::fmat& padded_input_channel = padded_input->slice(input_c);
        uint32_t tile_x = tile_ind_x * 4;
        uint32_t tile_y = tile_ind_y * 4;
        float in[6][6];
        for (int i = 0; i < 6; i++) {
          const float* col_ptr = padded_input_channel.colptr(tile_x + i);
          for (int j = 0; j < 6; j++) {
            in[i][j] = *(col_ptr + tile_y + j);
          }
        }

        arma::fmat& padded_out_channel = padded_out->slice(0);
        const arma::fmat& Y = Winograd4x3(channels_transform.at(input_c), in);
        for (int i = 0; i < 4; ++i) {
          float* out_col_ptr = padded_out_channel.colptr(tile_ind_x * 4 + i);
          for (int j = 0; j < 4; ++j) {
            *(out_col_ptr + tile_ind_y * 4 + j) += Y.at(j, i);
          }
        }
      }
    }
  }

  for (uint32_t w = 0; w < output_w; w++) {
    for (uint32_t h = 0; h < output_h; h++) {
      output->at(0, h, w) = padded_out->at(0, h, w);
    }
  }
}

}  // namespace kuiper_infer