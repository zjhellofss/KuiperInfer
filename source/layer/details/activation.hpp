//
// Created by fss on 23-12-14.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
#include "data/tensor.hpp"
#include "status_code.hpp"
namespace kuiper_infer {
namespace activation {
using ActivationFunc = std::function<void(sftensor, sftensor)>;

enum class ActivationType {
  kActivatetionUnknown = -1,
  kActivationRelu = 0,
  kActivationSilu = 1,
  kActivationSigmoid = 2,
  kActivationHardSwish = 3,
  kActivationHardSigmoid = 4,
  kActivationRelu6 = 5,
};

std::string ActivationTypeToString(ActivationType type);

StatusCode ActivationForward(ActivationType type,
                             const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                             std::vector<std::shared_ptr<Tensor<float>>>& outputs);
}  // namespace activation
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
