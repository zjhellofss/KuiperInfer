//
// Created by fss on 22-11-13.
//

#include "sigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

InferStatus SigmoidLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of sigmoid layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output size is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    CHECK(input == nullptr || !input->empty()) << "The input feature map of sigmoid layer is empty!";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->channels(), input->rows(), input->cols());
    }

    CHECK (output->shapes() == input->shapes()) << "The output size of sigmoid is error";

    output->set_data(input->data());
    arma::fcube &output_data_cube = output->data();
    output_data_cube = 1 / (1 + arma::exp(-output_data_cube));
    outputs.at(i) = output;
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus SigmoidLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                   std::shared_ptr<Layer> &sigmoid_layer) {
  CHECK(op != nullptr) << "Sigmoid operator is nullptr";
  sigmoid_layer = std::make_shared<SigmoidLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kSigmoidGetInstance("nn.Sigmoid", SigmoidLayer::GetInstance);
}