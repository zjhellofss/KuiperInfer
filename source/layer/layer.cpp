//
// Created by fss on 22-11-15.
//
#include "layer.hpp"
namespace kuiper_infer {

const std::vector<std::shared_ptr<Tensor>> &Layer::weights() const {
  LOG(FATAL) << "Layer not implement yet!";
}

const std::vector<std::shared_ptr<Tensor>> &Layer::bias() const {
  LOG(FATAL) << "Layer not implement yet!";
}

void Layer::set_bias(const std::vector<double> &bias) {
  LOG(FATAL) << "Layer not implement yet!";
}

void Layer::set_bias(const std::vector<std::shared_ptr<Tensor>> &bias) {
  LOG(FATAL) << "Layer not implement yet!";
}

void Layer::set_weights(const std::vector<double> &weights) {
  LOG(FATAL) << "Layer not implement yet!";
}

void Layer::set_weights(const std::vector<std::shared_ptr<Tensor>> &weights) {
  LOG(FATAL) << "Layer not implement yet!";
}

const std::string &Layer::layer_name() const {
  return this->layer_name_;
}

InferStatus Layer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs1,
                           const std::vector<std::shared_ptr<Tensor>> &inputs2,
                           std::vector<std::shared_ptr<Tensor>> &ouputs) {
  LOG(FATAL) << "Layer not implement yet!";
}

InferStatus Layer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                           std::vector<std::shared_ptr<Tensor>> &outputs) {
  LOG(FATAL) << "Layer not implement yet!";
}
}