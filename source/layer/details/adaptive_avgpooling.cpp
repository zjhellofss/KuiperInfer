//
// Created by fss on 22-11-12.
//
#include "adaptive_avgpooling.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

AdaptiveAveragePoolingLayer::AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w)
    : Layer("AdaptiveAveragePooling"), output_h_(output_h), output_w_(output_w) {
}

InferStatus AdaptiveAveragePoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                                 std::vector<std::shared_ptr<Tensor<float>>> &outputs) {

  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of average pooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output size is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  if (output_w_ <= 0 || output_h_ <= 0) {
    LOG(ERROR) << "The size of the output feature map is less than zero";
    return InferStatus::kInferFailedOutputSizeError;
  }

  const uint32_t batch = inputs.size();
#pragma omp parallel for num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
    CHECK(input_data == nullptr || !input_data->empty()) << "The input feature map of average pooling layer is empty";

    const uint32_t input_h = input_data->rows();
    const uint32_t input_w = input_data->cols();
    const uint32_t input_c = input_data->channels();
    const uint32_t stride_h = uint32_t(std::floor(input_h / output_h_));
    const uint32_t stride_w = uint32_t(std::floor(input_w / output_w_));
    CHECK(stride_w > 0 && stride_h > 0) << "The stride parameter is set incorrectly. It must always be greater than 0";

    const uint32_t pooling_h = input_h - (output_h_ - 1) * stride_h;
    const uint32_t pooling_w = input_w - (output_w_ - 1) * stride_w;
    CHECK(pooling_w > 0 && pooling_h > 0)
            << "The pooling parameter is set incorrectly. It must always be greater than 0";

    std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
    if (output_data == nullptr || output_data->empty()) {
      LOG(ERROR) << "The output size of adaptive pooling is empty";
      output_data = std::make_shared<Tensor<float>>(input_c, output_h_, output_w_);
    }

    CHECK (output_data->rows() == output_h_ && output_data->cols() == output_w_
               && output_data->channels() == input_c) << "The output size of adaptive pooling is error";

    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat &input_channel = input_data->at(ic);
      arma::fmat &output_channel = output_data->at(ic);
      for (uint32_t c = 0; c < input_w - pooling_w + 1; c += stride_w) {
        for (uint32_t r = 0; r < input_h - pooling_h + 1; r += stride_h) {

          float mean_value = 0.f;
          float *output_channel_ptr = output_channel.colptr(int(c / stride_w));
          for (uint32_t w = 0; w < pooling_w; ++w) {
            const float *col_ptr = input_channel.colptr(c + w) + r;
            for (uint32_t h = 0; h < pooling_h; ++h) {
              float current_value = *(col_ptr + h);
              mean_value = mean_value + current_value;
            }
          }
          *(output_channel_ptr + int(r / stride_h)) = mean_value / float(pooling_h * pooling_w);
        }
      }
    }
    outputs.at(i) = output_data;
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus AdaptiveAveragePoolingLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                                  std::shared_ptr<Layer> &avg_layer) {
  CHECK(op != nullptr) << "Adaptive pooling operator is nullptr";
  const auto &params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";

  const auto &output_hw = dynamic_cast<RuntimeParameterIntArray *>(params.at("output_size"));
  if (!output_hw) {
    LOG(ERROR) << "Can not find the output size parameter";
    return ParseParameterAttrStatus::kParameterMissingOutHW;
  }

  const auto &output_hw_arr = output_hw->value;
  if (output_hw_arr.size() != 2) {
    LOG(ERROR) << "Can not find the output size parameter";
    return ParseParameterAttrStatus::kParameterMissingOutHW;
  }

  avg_layer = std::make_shared<AdaptiveAveragePoolingLayer>(output_hw_arr.at(0), output_hw_arr.at(1));
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kAdaptiveAvgpoolingGetInstance("nn.AdaptiveAvgPool2d", AdaptiveAveragePoolingLayer::GetInstance);

}