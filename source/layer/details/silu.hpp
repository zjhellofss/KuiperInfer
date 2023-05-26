//
// Created by fss on 22-12-25.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class SiLULayer : public NonParamLayer {
 public:
  explicit SiLULayer();

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& silu_layer);
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
