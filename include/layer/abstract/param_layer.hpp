//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
#define KUIPER_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
#include "layer.hpp"

namespace kuiper_infer {
class ParamLayer : public Layer {
 public:
  explicit ParamLayer(const std::string &layer_name);

  /**
   * 初始化权重空间
   * @param param_count   卷积核数量
   * @param param_channel 卷积的通道数量
   * @param param_height  卷积的高度
   * @param param_width   卷积的宽度
   */
  void InitWeightParam(const uint32_t param_count, const uint32_t param_channel,
                       const uint32_t param_height, const uint32_t param_width);

  /**
   * 初始化偏移参数
   * @param param_count 偏移参数数量
   * @param param_channel 偏移参数通道数量
   * @param param_height 偏移参数高度
   * @param param_width  偏移参数宽度
   */
  void InitBiasParam(const uint32_t param_count, const uint32_t param_channel,
                     const uint32_t param_height, const uint32_t param_width);

  /**
   * 返回权重参数
   * @return 权重参数
   */
  const std::vector<std::shared_ptr<Tensor<float>>> &weights() const override;

  /**
   * 返回偏移参数
   * @return 偏移参数
   */
  const std::vector<std::shared_ptr<Tensor<float>>> &bias() const override;

  /**
   * 设置权重参数
   * @param weights 权重参数
   */
  void set_weights(const std::vector<float> &weights) override;

  /**
   * 设置偏移量参数
   * @param bias 偏移量参数
   */
  void set_bias(const std::vector<float> &bias) override;

  /**
   * 设置权重参数
   * @param weights 权重参数
   */
  void set_weights(
      const std::vector<std::shared_ptr<Tensor<float>>> &weights) override;

  /**
   * 设置偏移量参数
   * @param bias 偏移量参数
   */
  void set_bias(
      const std::vector<std::shared_ptr<Tensor<float>>> &bias) override;

 protected:
  std::vector<std::shared_ptr<Tensor<float>>> weights_;
  std::vector<std::shared_ptr<Tensor<float>>> bias_;
};

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
