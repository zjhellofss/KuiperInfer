//
// Created by fss on 22-11-15.
//
#include "layer.hpp"
namespace kuiper_infer {

const std::vector<std::shared_ptr<Blob>> &Layer::weights() const {
  LOG(FATAL) << "Layer not implement yet!";
}

const std::vector<std::shared_ptr<Blob>> &Layer::bias() const {
  LOG(FATAL) << "Layer not implement yet!";
}

void Layer::set_bias(const std::vector<double> &bias) {
  LOG(FATAL) << "Layer not implement yet!";
}

void Layer::set_bias(const std::vector<std::shared_ptr<Blob>> &bias) {
  LOG(FATAL) << "Layer not implement yet!";
}

void Layer::set_weights(const std::vector<double> &weights) {
  LOG(FATAL) << "Layer not implement yet!";
}

void Layer::set_weights(const std::vector<std::shared_ptr<Blob>> &weights) {
  LOG(FATAL) << "Layer not implement yet!";
}

}