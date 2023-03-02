//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#define KUIPER_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#include "layer/abstract/layer.hpp"

namespace kuiper_infer {
class SoftmaxLayer : public Layer {
 public:
  explicit SoftmaxLayer();

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

};
}

#endif //KUIPER_INFER_SOURCE_LAYER_SOFTMAX_HPP_
