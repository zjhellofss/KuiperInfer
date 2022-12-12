//
// Created by fss on 22-11-18.
//
#include "relu.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {
InferStatus ReluLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                               std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of relu layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    CHECK(!input->empty()) << "Relu layer input is empty";
    const std::shared_ptr<Tensor<float>> &output = outputs.at(i);
    output->set_data(input->data());
    output->Transform([](float val) {
      return val > 0. ? val : 0.;
    });
    outputs.at(i) = output;
  }

  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus ReluLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                std::shared_ptr<Layer> &relu_layer) {
  CHECK(op != nullptr) << "Relu operator is empty";
  relu_layer = std::make_shared<ReluLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kReluGetInstance("nn.ReLU", ReluLayer::GetInstance);
}

