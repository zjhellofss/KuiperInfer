//
// Created by fss on 22-12-12.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
#include "layer/abstract/layer.hpp"

#endif //KUIPER_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
namespace kuiper_infer {
class HardSwishLayer : public Layer {
 public:
  explicit HardSwishLayer();

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &hardswish_layer);
};
}