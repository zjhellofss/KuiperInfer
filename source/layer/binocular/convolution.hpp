//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_CONVOLUTION_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_CONVOLUTION_HPP_
#include "layer/abstract/param_layer.hpp"

namespace kuiper_infer {
class ConvolutionLayer : public ParamLayer {
 public:
  explicit ConvolutionLayer(uint32_t output_channel, uint32_t in_channel, uint32_t kernel_h,
                            uint32_t kernel_w, uint32_t padding, uint32_t stride_h, uint32_t stride_w,
                            bool use_bias = true);

  static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &conv_layer);

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      std::vector<std::shared_ptr<Tensor>> &outputs) override;

 private:
  bool use_bias_ = false;
  uint32_t padding_ = 0;
  uint32_t stride_h_ = 1;
  uint32_t stride_w_ = 1;
};

}

#endif //KUIPER_COURSE_SOURCE_LAYER_CONVOLUTION_HPP_
