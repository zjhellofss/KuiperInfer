//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_AVGPOOLING_HPP_
#define KUIPER_INFER_SOURCE_LAYER_AVGPOOLING_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class AdaptiveAveragePoolingLayer : public NonParamLayer {
 public:
  explicit AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& avg_layer);

 private:
  uint32_t output_h_ = 0;
  uint32_t output_w_ = 0;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_AVGPOOLING_HPP_
