//
// Created by fushenshen on 2023/3/15.
//
#include "winograd.hpp"
#include <glog/logging.h>
#include "tick.hpp"
namespace kuiper_infer {

void WinogradTransformG(const arma::fmat& g, arma::fmat& transform_g) {
  arma::fmat Gg(6, 3);
  for (int i = 0; i < 3; ++i) {
    float g0 = g.at(i, 0);
    float g1 = g.at(i, 1);
    float g2 = g.at(i, 2);
    float* Gg_colptr = Gg.colptr(i);

    *(Gg_colptr + 0) = g0 / 4.f;
    *(Gg_colptr + 1) = ((-g0 - g1 - g2) / 2.f) / 3.f;
    *(Gg_colptr + 2) = ((-g0 + g1 - g2) / 2.f) / 3.f;
    *(Gg_colptr + 3) = (g0 / 8.f + g1 / 4.f + g2 / 2.f) / 3.f;
    *(Gg_colptr + 4) = (g0 / 8.f - g1 / 4.f + g2 / 2.f) / 3.f;
    *(Gg_colptr + 5) = g2;
  }

  const arma::fmat& Gg_t = Gg.t();
  transform_g = arma::fmat(6, 6);
  for (int i = 0; i < 6; ++i) {
    float* Gg_t_colptr = (float*)Gg_t.colptr(i);
    const float Gg0 = *(Gg_t_colptr + 0);
    const float Gg1 = *(Gg_t_colptr + 1);
    const float Gg2 = *(Gg_t_colptr + 2);

    transform_g.at(0, i) = Gg0 / 4.f;
    transform_g.at(1, i) = ((-Gg0 - Gg1 - Gg2) / 2.f) / 3.f;
    transform_g.at(2, i) = ((-Gg0 + Gg1 - Gg2) / 2.f) / 3.f;
    transform_g.at(3, i) = (Gg0 / 8.f + Gg1 / 4.f + Gg2 / 2.f) / 3.f;
    transform_g.at(4, i) = (Gg0 / 8.f - Gg1 / 4.f + Gg2 / 2.f) / 3.f;
    transform_g.at(5, i) = Gg2;
  }
}

void Winograd4x32(const arma::fmat& transform_g, const std::vector<float>& d,
                  std::vector<float>& Y) {
  CHECK_EQ(d.size(), 36);
  CHECK_EQ(transform_g.empty(), false);
  CHECK(transform_g.n_cols == 6 && transform_g.n_rows == 6);

  float BTd[6][6];
  for (int i = 0; i < 6; ++i) {
    const float d0 = d[0 * 6 + i];
    const float d1 = d[1 * 6 + i];
    const float d2 = d[2 * 6 + i];
    const float d3 = d[3 * 6 + i];
    const float d4 = d[4 * 6 + i];
    const float d5 = d[5 * 6 + i];

    BTd[0][i] = (d0 * 4) - d2 * 5 + d4;
    BTd[1][i] = -(d1 * 4) - d2 * 4 + d3 + d4;
    BTd[2][i] = (d1 * 4) - d2 * 4 - d3 + d4;
    BTd[3][i] = -(d1 * 2) - d2 + d3 * 2 + d4;
    BTd[4][i] = (d1 * 2) - d2 - d3 * 2 + d4;
    BTd[5][i] = (d1 * 4) - d3 * 5 + d5;
  }

  arma::fmat V(6, 6);
  for (int i = 0; i < 6; ++i) {
    const float BTd0 = BTd[i][0];
    const float BTd1 = BTd[i][1];
    const float BTd2 = BTd[i][2];
    const float BTd3 = BTd[i][3];
    const float BTd4 = BTd[i][4];
    const float BTd5 = BTd[i][5];
    V.at(i, 0) = ((BTd0 * 4) - (BTd2 * 4) - BTd2 + BTd4) * transform_g.at(0, i);
    V.at(i, 1) =
        (-(BTd1 * 4) - (BTd2 * 4) + BTd3 + BTd4) * transform_g.at(1, i);
    V.at(i, 2) = ((BTd1 * 4) - (BTd2 * 4) - BTd3 + BTd4) * transform_g.at(2, i);
    V.at(i, 3) = (-(BTd1 * 2) - BTd2 + BTd3 * 2 + BTd4) * transform_g.at(3, i);
    V.at(i, 4) = ((BTd1 * 2) - BTd2 - BTd3 * 2 + BTd4) * transform_g.at(4, i);
    V.at(i, 5) = ((BTd1 * 4) - (BTd3 * 4) - BTd3 + BTd5) * transform_g.at(5, i);
  }

  float ATM[4][6];
  for (int i = 0; i < 6; ++i) {
    float* M_colptr = (float*)V.colptr(i);
    float M0 = *(M_colptr + 0);
    float M1 = *(M_colptr + 1);
    float M2 = *(M_colptr + 2);
    float M3 = *(M_colptr + 3);
    float M4 = *(M_colptr + 4);
    float M5 = *(M_colptr + 5);

    ATM[0][i] = M0 + M1 + M2 + M3 + M4;
    ATM[1][i] = M1 - M2 + (M3 * 2) - (M4 * 2);
    ATM[2][i] = M1 + M2 + (M3 * 4) + (M4 * 4);
    ATM[3][i] = M1 - M2 + (M3 * 8) - (M4 * 8) + M5;
  }

  for (int i = 0; i < 4; ++i) {
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

  const uint32_t output_h = (input_h - ((kernel_h - 1) + 1)) / stride_h + 1;
  const uint32_t output_w = (input_w - ((kernel_w - 1) + 1)) / stride_w + 1;
  if (output == nullptr || output->empty()) {
    output = std::make_shared<ftensor>(kernel_count, output_h, output_w);
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
  sftensor padded_out =
      std::make_shared<ftensor>(kernel_count, padded_out_h, padded_out_w);

  // 填充数据
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

#pragma omp parallel for
  for (uint32_t k = 0; k < kernel_count; ++k) {
    const auto& weight = weights.at(k);
    CHECK_EQ(weight->rows(), kernel_h);
    CHECK_EQ(weight->cols(), kernel_w);
    CHECK_EQ(weight->channels(), input_channels);
    const arma::fmat& padded_out_channel = padded_out->slice(k);

    for (uint32_t input_c = 0; input_c < input_channels; ++input_c) {
      arma::fmat transform_g;
      WinogradTransformG(weight->slice(input_c), transform_g);
      const arma::fmat& padded_input_channel = padded_input->slice(input_c);

#pragma omp parallel for collapse(2)
      for (uint32_t tile_ind_x = 0; tile_ind_x < tile_num_w; ++tile_ind_x) {
        for (uint32_t tile_ind_y = 0; tile_ind_y < tile_num_h; ++tile_ind_y) {
          uint32_t tile_x = tile_ind_x * 4;
          uint32_t tile_y = tile_ind_y * 4;
          std::vector<float> in(36);
          for (int i = 0; i < 6; i++) {
            const float* col_ptr = padded_input_channel.colptr(tile_x + i);
            for (int j = 0; j < 6; j++) {
              in[i * 6 + j] = *(col_ptr + tile_y + j);
            }
          }

          std::vector<float> Y(16);
          Winograd4x32(transform_g, in, Y);
          for (int i = 0; i < 4; ++i) {
            const float* out_col_ptr =
                (padded_out_channel.colptr(tile_ind_x * 4 + i));
            for (int j = 0; j < 4; ++j) {
              *((float*)out_col_ptr + tile_ind_y * 4 + j) += Y[i * 4 + j];
            }
          }
        }
      }
    }

    arma::fmat& out_channel = output->slice(k);
    for (uint32_t w = 0; w < output_w; w++) {
      float* out_channel_ptr = out_channel.colptr(w);
      const float* padded_out_channel_ptr = padded_out_channel.colptr(w);
      memcpy(out_channel_ptr, padded_out_channel_ptr, sizeof(float) * output_h);
    }
  }
}

}  // namespace kuiper_infer