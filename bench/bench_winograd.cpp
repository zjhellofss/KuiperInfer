//
// Created by fss on 22-12-5.
//
#include <benchmark/benchmark.h>
#include <armadillo>
#include <glog/logging.h>
#include "../source/layer/details/convolution_3x3.hpp"

//static void BM_ConvWinograd1(benchmark::State &state) {
//  float g1[] = {1, 2, 3,
//                4, 5, 6,
//                7, 8, 9};
//  float d1[] = {1, 2, 3, 4,
//                5, 6, 7, 8,
//                9, 10, 11, 12,
//                13, 14, 15, 16};
//
//  std::vector<std::vector<arma::fmat>> g_arr_c1;
//  std::vector<std::vector<arma::fmat>> d_arr_c1;
//
//  int in_channels = 3;
//  int out_channels = 64;
//
//  for (int i = 0; i < 64 * 64; ++i) {
//    std::vector<arma::fmat> g_arr;
//    std::vector<arma::fmat> d_arr;
//
//    for (int c = 0; c < out_channels; ++c) {
//      arma::fmat g(g1, 3, 3);
//      arma::fmat d(d1, 4, 4);
//      g_arr.push_back(g);
//      d_arr.push_back(d);
//    }
//    g_arr_c1.push_back(g_arr);
//    d_arr_c1.push_back(d_arr);
//  }
//  int batch_size = 8;
//  for (auto _ : state) {
//    for (int b = 0; b < batch_size; ++b) {
//#pragma omp parallel for num_threads(64)
//      for (int i = 0; i < 64 * 64; ++i) {
//        const std::vector<arma::fmat> &g_tmp_arr = g_arr_c1.at(i);
//        const std::vector<arma::fmat> &d_tmp_arr = d_arr_c1.at(i);
//
//        arma::fmat GgGT(4, 4);
//        arma::fmat result(1, 4);
//#pragma omp parallel for num_threads(32)
//        for (int o = 0; o < out_channels; ++o) {
//          const arma::fmat &g_tmp = g_tmp_arr.at(o);
//          WinogradTransformG(g_tmp, GgGT);
//          arma::fmat result_sum(1, 4);
//          for (int c = 0; c < in_channels; ++c) {
//            const arma::fmat &d_tmp = d_tmp_arr.at(c);
//            Winograd(GgGT, d_tmp, result);
//            result_sum += result;
//          }
//        }
//      }
//    }
//  }
//}
//
//static void BM_ConvWinograd2(benchmark::State &state) {
//  float g1[] = {1, 2, 3,
//                4, 5, 6,
//                7, 8, 9};
//  float d1[] = {1, 2, 3, 4,
//                5, 6, 7, 8,
//                9, 10, 11, 12,
//                13, 14, 15, 16};
//
//  std::vector<std::vector<arma::fmat>> g_arr_c1;
//  std::vector<std::vector<arma::fmat>> d_arr_c1;
//
//  int in_channels = 64;
//  int out_channels = 128;
//
//  for (int i = 0; i < 32 * 32; ++i) {
//    std::vector<arma::fmat> g_arr;
//    std::vector<arma::fmat> d_arr;
//
//    for (int c = 0; c < out_channels; ++c) {
//      arma::fmat g(g1, 3, 3);
//      arma::fmat d(d1, 4, 4);
//      g_arr.push_back(g);
//      d_arr.push_back(d);
//    }
//    g_arr_c1.push_back(g_arr);
//    d_arr_c1.push_back(d_arr);
//  }
//  int batch_size = 8;
//  for (auto _ : state) {
//    for (int b = 0; b < batch_size; ++b) {
//#pragma omp parallel for num_threads(64)
//      for (int i = 0; i < 32 * 32; ++i) {
//        const std::vector<arma::fmat> &g_tmp_arr = g_arr_c1.at(i);
//        const std::vector<arma::fmat> &d_tmp_arr = d_arr_c1.at(i);
//
//        arma::fmat GgGT(4, 4);
//        arma::fmat result(1, 4);
//#pragma omp parallel for num_threads(32)
//        for (int o = 0; o < out_channels; ++o) {
//          const arma::fmat &g_tmp = g_tmp_arr.at(o);
//          WinogradTransformG(g_tmp, GgGT);
//          arma::fmat result_sum(1, 4);
//          for (int c = 0; c < in_channels; ++c) {
//            const arma::fmat &d_tmp = d_tmp_arr.at(c);
//            Winograd(GgGT, d_tmp, result);
//            result_sum += result;
//          }
//        }
//      }
//    }
//  }
//}
//
//static void BM_ConvWinograd3(benchmark::State &state) {
//  float g1[] = {1, 2, 3,
//                4, 5, 6,
//                7, 8, 9};
//  float d1[] = {1, 2, 3, 4,
//                5, 6, 7, 8,
//                9, 10, 11, 12,
//                13, 14, 15, 16};
//
//  std::vector<std::vector<arma::fmat>> g_arr_c1;
//  std::vector<std::vector<arma::fmat>> d_arr_c1;
//
//  int in_channels = 128;
//  int out_channels = 256;
//
//  for (int i = 0; i < 16 * 16; ++i) {
//    std::vector<arma::fmat> g_arr;
//    std::vector<arma::fmat> d_arr;
//
//    for (int c = 0; c < out_channels; ++c) {
//      arma::fmat g(g1, 3, 3);
//      arma::fmat d(d1, 4, 4);
//      g_arr.push_back(g);
//      d_arr.push_back(d);
//    }
//    g_arr_c1.push_back(g_arr);
//    d_arr_c1.push_back(d_arr);
//  }
//  int batch_size = 8;
//  for (auto _ : state) {
//    for (int b = 0; b < batch_size; ++b) {
//#pragma omp parallel for num_threads(64)
//      for (int i = 0; i < 16 * 16; ++i) {
//        const std::vector<arma::fmat> &g_tmp_arr = g_arr_c1.at(i);
//        const std::vector<arma::fmat> &d_tmp_arr = d_arr_c1.at(i);
//
//        arma::fmat GgGT(4, 4);
//        arma::fmat result(1, 4);
//#pragma omp parallel for num_threads(32)
//        for (int o = 0; o < out_channels; ++o) {
//          const arma::fmat &g_tmp = g_tmp_arr.at(o);
//          WinogradTransformG(g_tmp, GgGT);
//          arma::fmat result_sum(1, 4);
//          for (int c = 0; c < in_channels; ++c) {
//            const arma::fmat &d_tmp = d_tmp_arr.at(c);
//            Winograd(GgGT, d_tmp, result);
//            result_sum += result;
//          }
//        }
//      }
//    }
//  }
//}
//
//static void BM_ConvWinograd4(benchmark::State &state) {
//  float g1[] = {1, 2, 3,
//                4, 5, 6,
//                7, 8, 9};
//  float d1[] = {1, 2, 3, 4,
//                5, 6, 7, 8,
//                9, 10, 11, 12,
//                13, 14, 15, 16};
//
//  std::vector<std::vector<arma::fmat>> g_arr_c1;
//  std::vector<std::vector<arma::fmat>> d_arr_c1;
//
//  int in_channels = 256;
//  int out_channels = 512;
//
//  for (int i = 0; i < 8 * 8; ++i) {
//    std::vector<arma::fmat> g_arr;
//    std::vector<arma::fmat> d_arr;
//
//    for (int c = 0; c < out_channels; ++c) {
//      arma::fmat g(g1, 3, 3);
//      arma::fmat d(d1, 4, 4);
//      g_arr.push_back(g);
//      d_arr.push_back(d);
//    }
//    g_arr_c1.push_back(g_arr);
//    d_arr_c1.push_back(d_arr);
//  }
//  int batch_size = 8;
//  for (auto _ : state) {
//    for (int b = 0; b < batch_size; ++b) {
//#pragma omp parallel for num_threads(64)
//      for (int i = 0; i < 8 * 8; ++i) {
//        const std::vector<arma::fmat> &g_tmp_arr = g_arr_c1.at(i);
//        const std::vector<arma::fmat> &d_tmp_arr = d_arr_c1.at(i);
//
//        arma::fmat GgGT(4, 4);
//        arma::fmat result(1, 4);
//#pragma omp parallel for num_threads(32)
//        for (int o = 0; o < out_channels; ++o) {
//          const arma::fmat &g_tmp = g_tmp_arr.at(o);
//          WinogradTransformG(g_tmp, GgGT);
//          arma::fmat result_sum(1, 4);
//          for (int c = 0; c < in_channels; ++c) {
//            const arma::fmat &d_tmp = d_tmp_arr.at(c);
//            Winograd(GgGT, d_tmp, result);
//            result_sum += result;
//          }
//        }
//      }
//    }
//  }
//}


static void BM_ConvWinograd4(benchmark::State &state) {
  using namespace arma;

  arma::fmat input(226, 226);
  input.fill(1.);
  arma::fmat kernel1 = "1 2 3 ;"
                       "4 5 6 ;"
                       "7 8 9";
  arma::fmat kernel2(3, 3, arma::fill::ones);
  arma::fmat kernel3(3, 3, arma::fill::ones);

  uint32_t h = input.n_rows;
  uint32_t w = input.n_cols;
  uint32_t kh = kernel1.n_rows;
  uint32_t kw = kernel1.n_cols;
  uint32_t input_channels = 3;
  uint32_t output_channels = 64;

  arma::fcube input_cube(h, w, input_channels);

  arma::fcube kernel_cube(kh, kw, input_channels);
  std::vector<arma::fcube> kernel_cubes;

  for (uint32_t i = 0; i < input_channels; ++i) {
    input_cube.slice(i) = input;
  }
  kernel_cube.slice(0) = kernel1;
  kernel_cube.slice(1) = kernel2;
  kernel_cube.slice(2) = kernel3;


  for (uint32_t i = 0; i < output_channels; ++i) {
    kernel_cubes.push_back(kernel_cube);
  }

  const uint32_t tiles = 4;
  input_cube.insert_rows(input_cube.n_rows, tiles - input_cube.n_rows % tiles);
  input_cube.insert_cols(input_cube.n_cols, tiles - input_cube.n_cols % tiles);

  uint32_t h_o = input_cube.n_rows - kh + 1;
  uint32_t w_o = input_cube.n_cols - kw + 1;

  uint32_t h_o_ = h - kh + 1;
  uint32_t w_o_ = w - kw + 1;

  uint32_t h_oxh_w = int(h_o / 2 * w_o / 2);
  int batch_size = 8;
  for (auto _ : state) {
    arma::fcube outputs(h_o, w_o, output_channels);
#pragma omp parallel for num_threads(output_channels)
    for (int j = 0; j < batch_size; ++j) {
#pragma omp parallel for num_threads(output_channels)
      for (uint32_t o = 0; o < output_channels; ++o) {
        const arma::fcube &kernel_ = kernel_cubes.at(o);
        arma::fmat output = outputs.slice(o);

#pragma omp parallel for num_threads(input_channels)
        for (uint32_t i = 0; i < input_channels; ++i) {
          const arma::fmat &kernel_channel_ = kernel_.slice(i);
          const arma::fmat &input_channel = input_cube.slice(i);

          const arma::fmat &G1 = WinogradTransformG(kernel_channel_);
#pragma omp parallel for num_threads(h_oxh_w)
          for (uint32_t b = 0; b < h_oxh_w; ++b) {
            uint32_t r = b / (w_o / 2);
            uint32_t c = b % (w_o / 2);

            arma::fmat tile(tiles, tiles);
            for (uint32_t t = 0; t < tiles; ++t) {
              const float *mem_ptr = input_channel.colptr(2 * c + t) + 2 * r;
              memcpy(tile.memptr() + t * tiles, mem_ptr, sizeof(float) * tiles);
            }

            const arma::fmat &result = Winograd(G1, tile);
            const uint32_t half_tiles = tiles / 2;

#pragma gcc unroll 2
            for (int t = 0; t < half_tiles; ++t) {
              float *output_ptr = output.colptr(2 * c + t) + 2 * r;
#pragma gcc unroll 2
              for (int k = 0; k < half_tiles; ++k) {
                *(output_ptr + k) += *(result.colptr(t) + k);
              }
            }
          }
        }
        if (output.n_rows != h_o_ || output.n_cols != w_o_) {
          output = output.submat(0, 0, h_o_ - 1, w_o_ - 1);
        }
      }
    }
  }
}

static void BM_ConvWinograd5(benchmark::State &state) {
  using namespace arma;

  arma::fmat input(226, 226);
  input.fill(1.);
  arma::fmat kernel1 = "1 2 3 ;"
                       "4 5 6 ;"
                       "7 8 9";
  arma::fmat kernel2(3, 3, arma::fill::ones);
  arma::fmat kernel3(3, 3, arma::fill::ones);

  uint32_t h = input.n_rows;
  uint32_t w = input.n_cols;
  uint32_t kh = kernel1.n_rows;
  uint32_t kw = kernel1.n_cols;
  uint32_t input_channels = 3;
  uint32_t output_channels = 64;

  arma::fcube input_cube(h, w, input_channels);

  arma::fcube kernel_cube(kh, kw, input_channels);
  std::vector<arma::fcube> kernel_cubes;

  for (uint32_t i = 0; i < input_channels; ++i) {
    input_cube.slice(i) = input;
  }
  kernel_cube.slice(0) = kernel1;
  kernel_cube.slice(1) = kernel2;
  kernel_cube.slice(2) = kernel3;


  for (uint32_t i = 0; i < output_channels; ++i) {
    kernel_cubes.push_back(kernel_cube);
  }

  const uint32_t tiles = 4;
  input_cube.insert_rows(input_cube.n_rows, tiles - input_cube.n_rows % tiles);
  input_cube.insert_cols(input_cube.n_cols, tiles - input_cube.n_cols % tiles);

  uint32_t h_o = input_cube.n_rows - kh + 1;
  uint32_t w_o = input_cube.n_cols - kw + 1;

  uint32_t h_o_ = h - kh + 1;
  uint32_t w_o_ = w - kw + 1;

  uint32_t h_oxh_w = int(h_o / 2 * w_o / 2);
  int batch_size = 8;
  for (auto _ : state) {
    arma::fcube outputs(h_o, w_o, output_channels);
#pragma omp parallel for num_threads(output_channels)
    for (int j = 0; j < batch_size; ++j) {
#pragma omp parallel for num_threads(output_channels)
      for (uint32_t o = 0; o < output_channels; ++o) {
        const arma::fcube &kernel_ = kernel_cubes.at(o);
        arma::fmat output = outputs.slice(o);

#pragma omp parallel for num_threads(input_channels)
        for (uint32_t i = 0; i < input_channels; ++i) {
          const arma::fmat &kernel_channel_ = kernel_.slice(i);
          const arma::fmat &input_channel = input_cube.slice(i);

          const arma::fmat &G1 = WinogradTransformG(kernel_channel_);
#pragma omp parallel for num_threads(h_oxh_w)
          for (uint32_t b = 0; b < h_oxh_w; ++b) {
            uint32_t r = b / (w_o / 2);
            uint32_t c = b % (w_o / 2);

            arma::fmat tile(tiles, tiles);
            for (uint32_t t = 0; t < tiles; ++t) {
              const float *mem_ptr = input_channel.colptr(2 * c + t) + 2 * r;
              memcpy(tile.memptr() + t * tiles, mem_ptr, sizeof(float) * tiles);
            }

            const arma::fmat &result = Winograd(G1, tile);
            const uint32_t half_tiles = tiles / 2;

#pragma gcc unroll 2
            for (int t = 0; t < half_tiles; ++t) {
              float *output_ptr = output.colptr(2 * c + t) + 2 * r;
#pragma gcc unroll 2
              for (int k = 0; k < half_tiles; ++k) {
                *(output_ptr + k) += *(result.colptr(t) + k);
              }
            }
          }
        }
        if (output.n_rows != h_o_ || output.n_cols != w_o_) {
          output = output.submat(0, 0, h_o_ - 1, w_o_ - 1);
        }
      }
    }
  }
}

BENCHMARK(BM_ConvWinograd5)->Iterations(15);
