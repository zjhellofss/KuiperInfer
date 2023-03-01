//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_SIGMOID_HPP_
#define KUIPER_INFER_SOURCE_LAYER_SIGMOID_HPP_
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {
class SigmoidLayer : public Layer {
 public:
  explicit SigmoidLayer(): Layer("Sigmoid"){

  }
  InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                        std::shared_ptr<Layer> &sigmoid_layer);
};
}

#endif //KUIPER_INFER_SOURCE_LAYER_SIGMOID_HPP_
