//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#define KUIPER_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#include "layer/abstract/layer.hpp"

namespace kuiper_infer {
class SoftmaxLayer : public Layer {
 public:
  explicit SoftmaxLayer(int dim = -1);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& softmax_layer);

 private:
  int softmax_dim_ = -1;
};
}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_SOFTMAX_HPP_
