//
// Created by fss on 22-12-4.
//
#include "convolution_3x3.hpp"
#include <armadillo>
#include <glog/logging.h>
#include <x86intrin.h>

void WinogradTransformG(const arma::fmat &kernel, arma::fmat &kernel_g) {
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
  kernel_g = GT * Gg;
}

void Winograd(const arma::fmat &kernel_g, const arma::fmat &feature, arma::fmat &result) {
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

  result[0] = (AT_UV0_ + AT_UV1_ + AT_UV2);
  result[2] = (AT_UV4 + AT_UV5 + AT_UV6);
  result[1] = (AT_UV1_ - AT_UV2 - AT_UV3);
  result[3] = (AT_UV5 - AT_UV6 - AT_UV7);
}
