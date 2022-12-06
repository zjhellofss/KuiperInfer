//
// Created by fss on 22-12-4.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_CONVOLUTION_3X3_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_CONVOLUTION_3X3_HPP_
#include <cassert>
#include <armadillo>

void WinogradTransformG(const arma::fmat &kernel, arma::fmat &kernel_g);

void Winograd(const arma::fmat &kernel_g, const arma::fmat &feature, arma::fmat &result);
#endif //KUIPER_INFER_SOURCE_LAYER_DETAILS_CONVOLUTION_3X3_HPP_
