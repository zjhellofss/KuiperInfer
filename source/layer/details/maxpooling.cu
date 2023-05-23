//
// Created by fss on 22-11-18.
//

#include "maxpooling.hpp"
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
namespace kuiper_infer {


__global__ void MaxPoolForward(const int nthreads,
    const float* const input,float* const output, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w) {
 
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    const float* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

MaxPoolingLayer::MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w,
                                 uint32_t pooling_size_h,
                                 uint32_t pooling_size_w, uint32_t stride_h,
                                 uint32_t stride_w)
    : Layer("MaxPooling"),
      padding_h_(padding_h),
      padding_w_(padding_w),
      pooling_size_h_(pooling_size_h),
      pooling_size_w_(pooling_size_w),
      stride_h_(stride_h),
      stride_w_(stride_w) {}

InferStatus MaxPoolingLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the max pooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR)
        << "The input and output tensor array size of the max pooling layer "
           "do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch = inputs.size();
  const uint32_t pooling_h = pooling_size_h_;
  const uint32_t pooling_w = pooling_size_w_;
  if (!stride_h_ || !stride_w_) {
    LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                  "greater than 0";
    return InferStatus::kInferFailedStrideParameterError;
  }

  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<ftensor>& input_data = inputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input tensor array in the max pooling layer has an "
                    "empty tensor "
                 << i << "th";
      return InferStatus::kInferFailedInputEmpty;
    } else {
      uint32_t input_h = input_data->rows();
      uint32_t input_w = input_data->cols();
      uint32_t output_h = uint32_t(std::floor(
          (int(input_h) - int(pooling_h) + 2 * padding_h_) / stride_h_ + 1));
      uint32_t output_w = uint32_t(std::floor(
          (int(input_w) - int(pooling_w) + 2 * padding_w_) / stride_w_ + 1));
      if (!output_w || !output_h) {
        LOG(ERROR) << "The output size of tensor " << i << "th"
                   << " in the max pooling layer is less than zero";
        return InferStatus::kInferFailedOutputSizeError;
      } else {
        const std::shared_ptr<ftensor>& output_data = outputs.at(i);
        if (output_data != nullptr && !output_data->empty()) {
          if (output_data->rows() != output_h ||
              output_data->cols() != output_w) {
            LOG(ERROR) << "The output tensor array in the max pooling layer "
                          "has an incorrectly sized tensor "
                       << i << "th";
            return InferStatus::kInferFailedOutputSizeError;
          }
        }
      }
    }
  }







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