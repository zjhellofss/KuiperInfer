//
// Created by fss on 22-12-4.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_CONVOLUTION_3X3_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_CONVOLUTION_3X3_HPP_
#include <cassert>
#include <armadillo>

arma::fmat WinogradTransformG(const arma::fmat &kernel);

arma::fmat Winograd(const arma::fmat &kernel_g, const arma::fmat &feature);
#endif //KUIPER_INFER_SOURCE_LAYER_DETAILS_CONVOLUTION_3X3_HPP_
