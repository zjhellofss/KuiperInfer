//
// Created by fss on 22-11-22.
//

#ifndef KUIPER_COURSE_INCLUDE_DATA_IM2COL_HPP_
#define KUIPER_COURSE_INCLUDE_DATA_IM2COL_HPP_
#include "data/tensor.hpp"
#include <armadillo>
#include <glog/logging.h>
namespace kuiper_infer {
arma::mat Im2Col(const std::shared_ptr<Tensor> &input_data, const std::shared_ptr<Tensor> &kernel,
                 const uint32_t stride_h, const uint32_t stride_w,
                 const uint32_t output_h, const uint32_t output_w);
}
#endif //KUIPER_COURSE_INCLUDE_DATA_IM2COL_HPP_
