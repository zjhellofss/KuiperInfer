//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
#define KUIPER_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
#include "layer/abstract/param_layer.hpp"

namespace kuiper_infer {
class ConvolutionLayer : public ParamLayer {
 public:
  explicit ConvolutionLayer(uint32_t output_channel, uint32_t in_channel,
                            uint32_t kernel_h, uint32_t kernel_w,
                            uint32_t padding_h, uint32_t padding_w,
                            uint32_t stride_h, uint32_t stride_w,
                            uint32_t groups, bool use_bias = true);

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& conv_layer);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

 private:
  bool use_bias_ = false;
  uint32_t groups_ = 1;
  uint32_t padding_h_ = 0;
  uint32_t padding_w_ = 0;
  uint32_t stride_h_ = 1;
  uint32_t stride_w_ = 1;
};

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
