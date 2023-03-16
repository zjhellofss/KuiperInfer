//
// Created by fushenshen on 2023/3/15.
//

#ifndef KUIPER_INFER_WINOGRAD_HPP
#define KUIPER_INFER_WINOGRAD_HPP
#include "data/tensor.hpp"
namespace kuiper_infer {
void Convolution3x3s1(const std::shared_ptr<Tensor<float>>& input,
                      std::shared_ptr<Tensor<float>>& output,
                      const std::vector<sftensor>& weights);
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_WINOGRAD_HPP
