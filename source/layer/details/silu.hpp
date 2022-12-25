//
// Created by fss on 22-12-25.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {
class SiLULayer : public Layer {
 public:
  explicit SiLULayer();

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
};
}
#endif //KUIPER_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
