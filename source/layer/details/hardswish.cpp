//
// Created by fss on 22-12-12.
//
#include "hardswish.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
HardSwishLayer::HardSwishLayer() : Layer("HardSwish") {}

InferStatus HardSwishLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of convolution layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  const uint32_t batch = inputs.size();
#pragma omp parallel for num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    CHECK(!input->empty()) << "HardSwish layer input is empty";
    const std::shared_ptr<Tensor<float>> &output = outputs.at(i);

    output->set_data(input->data());
    output->Transform([](float val) {
      if (val <= -3.f) {
        return 0.f;
      } else if (val >= 3.f) {
        return val;
      } else {
        return val * (val + 3) / 6;
      }
    });
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus HardSwishLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                     std::shared_ptr<Layer> &hardswish_layer) {
  CHECK(op != nullptr) << "HardSwishLayer operator is empty";
  hardswish_layer = std::make_shared<HardSwishLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kHardSwishGetInstance("nn.Hardswish", HardSwishLayer::GetInstance);

}