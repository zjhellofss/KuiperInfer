//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
#include "layer.hpp"

namespace kuiper_infer {
class ParamLayer : public Layer {
 public:
  explicit ParamLayer(const std::string &layer_name);

  void InitWeightParam(const uint32_t param_count, const uint32_t param_channel,
                       const uint32_t param_height, const uint32_t param_width);

  void InitBiasParam(const uint32_t param_count, const uint32_t param_channel,
                     const uint32_t param_height, const uint32_t param_width);

  const std::vector<std::shared_ptr<Tensor<float>>> &weights() const override;

  const std::vector<std::shared_ptr<Tensor<float>>> &bias() const override;

  void set_weights(const std::vector<float> &weights) override;

  void set_bias(const std::vector<float> &bias) override;

  void set_weights(
      const std::vector<std::shared_ptr<Tensor<float>>> &weights) override;

  void set_bias(
      const std::vector<std::shared_ptr<Tensor<float>>> &bias) override;

 protected:
  std::vector<std::shared_ptr<Tensor<float>>> weights_;
  std::vector<std::shared_ptr<Tensor<float>>> bias_;
};

}  // namespace kuiper_infer

#endif  // KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
