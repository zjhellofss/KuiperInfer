//
// Created by fss on 22-11-18.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {
class ReluLayer : public Layer {
 public:
  InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      std::vector<std::shared_ptr<Tensor>> &outputs) override;
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_BINOCULAR_RELU_HPP_
