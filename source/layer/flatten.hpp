//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_FLATTEN_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_FLATTEN_HPP_
#include "layer.hpp"
namespace kuiper_infer {
class FlattenLayer : public Layer {
 public:
  explicit FlattenLayer();
  InferStatus Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                      std::vector<std::shared_ptr<Blob>> &outputs) override;
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_FLATTEN_HPP_
