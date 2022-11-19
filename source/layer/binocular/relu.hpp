//
// Created by fss on 22-11-18.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {
class ReluLayer : public Layer {
 public:
  ReluLayer() : Layer("Relu") {

  }
  InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      std::vector<std::shared_ptr<Tensor>> &outputs) override;

  static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &relu_layer);
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_BINOCULAR_RELU_HPP_
