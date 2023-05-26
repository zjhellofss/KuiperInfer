//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_INFER_SOURCE_NONPRARM_LAYER_LAYER_HPP_
#define KUIPER_INFER_SOURCE_NONPRARM_LAYER_LAYER_HPP_
#include "layer.hpp"

namespace kuiper_infer {
class RuntimeOperator;
class NonParamLayer : public Layer {
 public:
  explicit NonParamLayer(std::string layer_name)
      : Layer(std::move(layer_name)) {}

  virtual ~NonParamLayer() = default;
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_NONPRARM_LAYER_LAYER_HPP_
