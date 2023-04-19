//
// Created by fss on 22-12-12.
//
#include "hardsigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
HardSigmoid::HardSigmoid() : Layer("HardSigmoid") {}

InferStatus HardSigmoid::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the hardsigmoid layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the hardsigmoid "
                  "layer do not match";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  const uint32_t batch = inputs.size();
#pragma omp parallel for num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input == nullptr || !input->empty())
        << "The input tensor array in the hardsigmoid layer has an "
           "empty tensor";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The output tensor array in the hardsigmoid layer has an "
                     "empty tensor";
      output = std::make_shared<Tensor<float>>(input->channels(), input->rows(),
                                               input->cols());
      outputs.at(i) = output;
    }

    CHECK(output->shapes() == input->shapes())
        << "The output and input shapes of the hardsigmoid layer do "
           "not match!";
    output->set_data(input->data());
    output->Transform([](float val) {
      if (val <= -3.f) {
        return 0.f;
      } else if (val >= 3.f) {
        return 1.f;
      } else {
        return val / 6.f + 0.5f;
      }
    });
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus HardSigmoid::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& hardsigmoid_layer) {
  CHECK(op != nullptr) << "HardSigmoid operator is nullptr";
  hardsigmoid_layer = std::make_shared<HardSigmoid>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kHardSigmoidGetInstance("nn.Hardsigmoid",
                                               HardSigmoid::GetInstance);

}  // namespace kuiper_infer
