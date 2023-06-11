//
// Created by fss on 22-11-18.
//

#include <float.h>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "maxpooling.hpp"
#include "runtime/runtime_ir.hpp"
#include "utils/gpu_utils.cuh"

namespace kuiper_infer {

__global__ void MaxPoolForward(const int nthreads, const float* const input,
                               float* const output, const int channels,
                               const int height, const int width,
                               const int output_h, const int output_w,
                               const int kernel_h, const int kernel_w,
                               const int stride_h, const int stride_w,
                               const int pad_h, const int pad_w) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % output_w;
    const int ph = (index / output_w) % output_h;
    const int c = (index / output_w / output_h) % channels;
    const int n = index / output_w / output_h / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    const float* const input_slice =
        input + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (input_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
        
          maxval = input_slice[maxidx];
        }
      }
    }
    output[index] = maxval;
  }
}

MaxPoolingLayer::MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w,
                                 uint32_t pooling_h, uint32_t pooling_w,
                                 uint32_t stride_h, uint32_t stride_w)
    : Layer("MaxPooling"),
      padding_h_(padding_h),
      padding_w_(padding_w),
      kernel_h_(pooling_h),
      kernel_w_(pooling_w),
      stride_h_(stride_h),
      stride_w_(stride_w) {}

InferStatus MaxPoolingLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the max pooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (!stride_h_ || !stride_w_) {
    LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                  "greater than 0";
    return InferStatus::kInferFailedStrideParameterError;
  }
  uint32_t input_h = inputs.at(0).get()->rows();
  uint32_t input_w = inputs.at(0).get()->cols();

  uint32_t output_h = uint32_t(std::floor(
      (int(input_h) - int(kernel_h_) + 2 * padding_h_) / stride_h_ + 1));
  uint32_t output_w = uint32_t(std::floor(
      (int(input_w) - int(kernel_w_) + 2 * padding_w_) / stride_w_ + 1));

  if (!output_w || !output_h) {
    LOG(ERROR) << "The output size of tensor "
               << " in the max pooling layer is less than zero";
    return InferStatus::kInferFailedOutputSizeError;
  }
  if (outputs.empty()) {
    // 如果不是从计算图读入会经过这一部分
    std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(
        inputs.at(0)->nums(), inputs.at(0)->channels(), output_h, output_w);
    outputs.push_back(output);
  } else {
    if (outputs.at(0)->rows() != output_h ||
        outputs.at(0)->cols() != output_w) {
      LOG(ERROR) << "The output size of tensor is wrong";
      return InferStatus::kInferFailedOutputSizeError;
    }
  }

  uint32_t count =
      inputs.at(0)->nums() * inputs.at(0)->channels() * output_h * output_w;

  MaxPoolForward<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
      count, inputs.at(0)->gpu_data(), outputs.at(0)->gpu_data(),
      inputs.at(0)->channels(), input_h, input_w, output_h, output_w,
      kernel_h_, kernel_w_, stride_h_, stride_w_, padding_h_, padding_w_);

  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus MaxPoolingLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& max_layer) {
  CHECK(op != nullptr) << "MaxPooling get instance failed, operator is nullptr";
  const std::map<std::string, RuntimeParameter*>& params = op->params;
  if (params.find("stride") == params.end()) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  const auto& stride =
      dynamic_cast<RuntimeParameterIntArray*>(params.at("stride"));
  if (!stride) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  if (params.find("padding") == params.end()) {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  const auto& padding =
      dynamic_cast<RuntimeParameterIntArray*>(params.at("padding"));
  if (!padding) {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Can not find the kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  const auto& kernel_size =
      dynamic_cast<RuntimeParameterIntArray*>(params.at("kernel_size"));
  if (!kernel_size) {
    LOG(ERROR) << "Can not find the kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }
  const auto& padding_values = padding->value;
  const auto& stride_values = stride->value;
  const auto& kernel_values = kernel_size->value;

  const uint32_t dims = 2;
  if (padding_values.size() != dims) {
    LOG(ERROR) << "Can not find the right padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  if (stride_values.size() != dims) {
    LOG(ERROR) << "Can not find the right stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  if (kernel_values.size() != dims) {
    LOG(ERROR) << "Can not find the right kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  max_layer = std::make_shared<MaxPoolingLayer>(
      padding_values.at(0), padding_values.at(1), kernel_values.at(0),
      kernel_values.at(1), stride_values.at(0), stride_values.at(1));

  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kMaxPoolingGetInstance("nn.MaxPool2d",
                                              MaxPoolingLayer::GetInstance);

}  // namespace kuiper_infer