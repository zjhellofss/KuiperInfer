//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_LINEAR_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_LINEAR_HPP_
#include "layer.hpp"
#include "param_layer.hpp"

namespace kuiper_infer {
class LinearLayer : public ParamLayer {
 public:
  explicit LinearLayer(const std::vector<std::shared_ptr<Blob>> &weights, const std::vector<std::shared_ptr<Blob>> &bias);

  InferStatus Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                      std::vector<std::shared_ptr<Blob>> &outputs) override;
};
}

#endif //KUIPER_COURSE_SOURCE_LAYER_LINEAR_HPP_
