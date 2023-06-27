//
// Created by fss on 22-11-12.
//
#include "adaptive_avgpooling.hpp"
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"
#include "utils/gpu_utils.cuh"

namespace kuiper_infer {

__global__ void AdaptiveAveragePoolForward(const int nthreads, const float* const input,
                                          float* const output, const int channels,
                                          const int input_h, const int input_w,
                                          const int output_h, const int output_w,
                                          const int kernel_h, const int kernel_w,
                                          const int stride_h, const int stride_w) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % output_w;
    const int ph = (index / output_w) % output_h;
    const int c = (index / output_w / output_h) % channels;
    const int n = index / output_w / output_h / channels;
    int hstart = ph * stride_h;
    int wstart = pw * stride_w;
    const int hend = hstart + kernel_h;
    const int wend = wstart + kernel_w;

    float avgval = 0;
    const float* const input_slice = 
        input + (n * channels + c) * input_h * input_w;
    
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        avgval += input_slice[h * input_w + w];
      }
    }
    const uint32_t kernel_size = kernel_h * kernel_w;
    output[index] = avgval / float(kernel_size);
  }
}

AdaptiveAveragePoolingLayer::AdaptiveAveragePoolingLayer(uint32_t output_h,
                                                         uint32_t output_w)
    : Layer("AdaptiveAveragePooling"),
      output_h_(output_h),
      output_w_(output_w) {}

InferStatus AdaptiveAveragePoolingLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR)
        << "The input tensor array in the adaptive pooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  uint32_t input_h = inputs.at(0).get()->rows();
  uint32_t input_w = inputs.at(0).get()->cols();

  uint32_t stride_h = uint32_t(std::floor(input_h / output_h_));
  uint32_t stride_w = uint32_t(std::floor(input_w / output_w_));
  CHECK(stride_w > 0 && stride_h > 0)
        << "The stride parameter is set incorrectly. It must always be greater "
           "than 0";

  uint32_t kernel_h = 
            (int)input_h - (int(output_h_) - 1) * int(stride_h);
  uint32_t kernel_w = 
            (int)input_w - (int(output_w_) - 1) - int(stride_w);

  CHECK(kernel_w > 0 && kernel_h > 0)
        << "The kerneling parameter is set incorrectly. It must always be "
           "greater than 0";

  if (!output_w_ || !output_h_) {
    LOG(ERROR) << "The output size of tensor "
               << " in the max pooling layer is less than zero";
    return InferStatus::kInferFailedOutputSizeError;
  }

  if (outputs.empty()) {
    DLOG(ERROR) << "The output tensor array in the adaptive pooling layer is empty";
    std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(
      inputs.at(0)->nums(), inputs.at(0)->channels(), output_h_, output_w_);
    outputs.push_back(output);
  } else {
  if (outputs.at(0)->rows() != output_h_ ||
        outputs.at(0)->cols() != output_w_) {
      LOG(ERROR) << "The output size of tensor is wrong";
      return InferStatus::kInferFailedOutputSizeError;
    }
  }

  // 计算一共需要多少个gpu线程
  uint32_t count =
      inputs.at(0)->nums() * inputs.at(0)->channels() * output_h_ * output_w_;
    
  AdaptiveAveragePoolForward<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
    count, inputs.at(0)->gpu_data(), outputs.at(0)->gpu_data(),
    inputs.at(0)->channels(), input_h, input_w, output_h_, output_w_,
    kernel_h, kernel_w, stride_h, stride_w);
    
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus AdaptiveAveragePoolingLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& avg_layer) {
  CHECK(op != nullptr) << "Adaptive pooling operator is nullptr";
  const auto& params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";

  const auto& output_hw =
      dynamic_cast<RuntimeParameterIntArray*>(params.at("output_size"));
  if (!output_hw) {
    LOG(ERROR) << "Can not find the output size parameter";
    return ParseParameterAttrStatus::kParameterMissingOutHW;
  }

  const auto& output_hw_arr = output_hw->value;
  if (output_hw_arr.size() != 2) {
    LOG(ERROR) << "Can not find the output size parameter";
    return ParseParameterAttrStatus::kParameterMissingOutHW;
  }
  avg_layer = std::make_shared<AdaptiveAveragePoolingLayer>(
      output_hw_arr.at(0), output_hw_arr.at(1));
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kAdaptiveAvgpoolingGetInstance(
    "nn.AdaptiveAvgPool2d", AdaptiveAveragePoolingLayer::GetInstance);

}  // namespace kuiper_infer