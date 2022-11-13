//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_SOFTMAX_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_SOFTMAX_HPP_
#include "layer.hpp"

namespace kuiper_infer {
class SoftmaxLayer : public Layer {
 public:
  explicit SoftmaxLayer();

  InferStatus Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                      std::vector<std::shared_ptr<Blob>> &outputs) override;

};
}

#endif //KUIPER_COURSE_SOURCE_LAYER_SOFTMAX_HPP_
