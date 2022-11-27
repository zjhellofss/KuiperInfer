//
// Created by fss on 22-11-27.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <vector>
#include "data/tensor.hpp"
#include "../source/layer/binocular/convolution.hpp"

TEST(test_im2col, conv_1x1) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 4, 4);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  arma::fmat data1("1  2  3  4 ;"
                   "5  6  7  8 ;"
                   "9  10 12 12;"
                   "13 14 15 16");
  input->at(0) = data1;

  std::shared_ptr<Tensor<float>> kernel = std::make_shared<Tensor<float>>(1, 2, 2);
  std::vector<std::shared_ptr<Tensor<float>>> kernels;

  arma::fmat data2("9  10;"
                   "11 12");
  kernel->at(0) = data2;

  kernels.push_back(kernel);
  kernels.push_back(kernel);
  kernels.push_back(kernel);


  ConvolutionLayer conv(1, 1, 2, 2, 0, 1, 1, false);
  conv.set_weights(kernels);

  conv.Forward(inputs, kernels);
}