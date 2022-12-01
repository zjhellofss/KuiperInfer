//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_LAYER_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_LAYER_HPP_
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <glog/logging.h>

#include "status_code.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_op.hpp"

namespace kuiper_infer {

class Layer {
 public:
  explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {

  }

  virtual ~Layer() = default;

  virtual InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                              std::vector<std::shared_ptr<Tensor<float>>> &outputs);

  virtual InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs1,
                              const std::vector<std::shared_ptr<Tensor<float>>> &inputs2,
                              std::vector<std::shared_ptr<Tensor<float>>> &ouputs);

  virtual const std::vector<std::shared_ptr<Tensor<float>>> &weights() const;

  virtual const std::vector<std::shared_ptr<Tensor<float>>> &bias() const;

  virtual void set_weights(const std::vector<std::shared_ptr<Tensor<float>>> &weights);

  virtual void set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias);

  virtual void set_weights(const std::vector<float> &weights);

  virtual void set_bias(const std::vector<float> &bias);

  virtual const std::string &layer_name() const { return this->layer_name_; }

 protected:
  std::string layer_name_;
};

}
#endif //KUIPER_COURSE_SOURCE_LAYER_LAYER_HPP_
