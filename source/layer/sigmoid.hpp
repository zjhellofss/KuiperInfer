//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_SIGMOID_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_SIGMOID_HPP_
#include "layer.hpp"
namespace kuiper_infer {
class SigmoidLayer : public Layer {
 public:
  SigmoidLayer(): Layer("Sigmoid"){

  }
  InferStatus Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                      std::vector<std::shared_ptr<Blob>> &outputs) override;
};
}

#endif //KUIPER_COURSE_SOURCE_LAYER_SIGMOID_HPP_
