//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_FLATTEN_HPP_
#define KUIPER_INFER_SOURCE_LAYER_FLATTEN_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class ViewLayer : public NonParamLayer {
 public:
  explicit ViewLayer(const std::vector<int32_t>& shapes);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& view_layer);

 private:
  std::vector<int32_t> shapes_;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_FLATTEN_HPP_
