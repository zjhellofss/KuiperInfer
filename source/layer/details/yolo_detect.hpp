//
// Created by fss on 22-12-26.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
#include "convolution.hpp"
#include "layer/abstract/layer.hpp"

namespace kuiper_infer {
class YoloDetectLayer : public Layer {
 public:
  explicit YoloDetectLayer(
      int32_t stages, int32_t num_classes, int32_t num_anchors,
      const std::vector<float>& strides,
      const std::vector<arma::fmat>& anchor_grids,
      const std::vector<arma::fmat>& grids,
      const std::vector<std::shared_ptr<ConvolutionLayer>>& conv_layers);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& yolo_detect_layer);

  void set_stage_tensors(const std::vector<sftensor>& stage_tensors);

 private:
  int32_t stages_ = 0;
  int32_t num_classes_ = 0;
  int32_t num_anchors_ = 0;
  std::vector<float> strides_;
  std::vector<arma::fmat> anchor_grids_;
  std::vector<arma::fmat> grids_;
  std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers_;
  std::vector<sftensor> stages_tensors_;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
