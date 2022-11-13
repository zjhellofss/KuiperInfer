//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
#include"layer.hpp"

namespace kuiper_infer {
class ParamLayer : public Layer {
 public:
  explicit ParamLayer(const std::string &layer_name, const std::vector<std::shared_ptr<Blob>> &weights, const std::vector<std::shared_ptr<Blob>> &bias);

  const std::vector<std::shared_ptr<Blob>> &weights() const;

  const std::vector<std::shared_ptr<Blob>> &bias() const;

  void set_weights(const std::vector<std::shared_ptr<Blob>> &weights);

  void set_bias(const std::vector<std::shared_ptr<Blob>> &bias);

  void set_weights(const std::vector<double> &weights);

  void set_bias(const std::vector<double> &bias);

  InferStatus Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                      std::vector<std::shared_ptr<Blob>> &outputs) override;

 protected:
  std::vector<std::shared_ptr<Blob>> weights_;
  std::vector<std::shared_ptr<Blob>> bias_;
};

}

#endif //KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
