//
// Created by fss on 22-11-18.
//

#include "maxpooling.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
namespace kuiper_infer {

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
    LOG(ERROR) << "The input feature map of max pooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR)
        << "The input and output size of max pooling layer is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
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
      LOG(ERROR) << "The input feature map of max pooling layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    } else {
      uint32_t input_h = input_data->rows();
      uint32_t input_w = input_data->cols();
      uint32_t output_h = uint32_t(
          std::floor((input_h - pooling_h + 2 * padding_h_) / stride_h_ + 1));
      uint32_t output_w = uint32_t(
          std::floor((input_w - pooling_w + 2 * padding_w_) / stride_w_ + 1));
      if (!output_w || !output_h) {
        LOG(ERROR) << "The output size of max pooling layer is less than zero";
        return InferStatus::kInferFailedOutputSizeError;
      } else {
        const std::shared_ptr<ftensor>& output_data = outputs.at(i);
        if (output_data != nullptr && !output_data->empty()) {
          if (output_data->rows() != output_h ||
              output_data->cols() != output_w) {
            LOG(ERROR)
                << "The output size of max pooling layer is not adapting";
            return InferStatus::kInferFailedOutputSizeError;
          }
        }
      }
    }
  }

#pragma omp parallel for num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
    std::shared_ptr<Tensor<float>> input_data_;
    CHECK(input_data == nullptr || !input_data->empty())
        << "The input feature map of max pooling layer is empty";

    if (padding_h_ > 0 || padding_w_ > 0) {
      //      input_data_ = input_data->Clone();
      //      input_data_->Padding({padding_h_, padding_h_, padding_w_,
      //      padding_w_}, std::numeric_limits<float>::lowest());
      input_data_ = TensorPadding(
          input_data, {padding_h_, padding_h_, padding_w_, padding_w_},
          std::numeric_limits<float>::lowest());
    } else {
      input_data_ = input_data;
    }

    const uint32_t input_h = input_data_->rows();
    const uint32_t input_w = input_data_->cols();
    const uint32_t input_c = input_data_->channels();

    const uint32_t output_h =
        uint32_t(std::floor((input_h - pooling_h) / stride_h_ + 1));
    const uint32_t output_w =
        uint32_t(std::floor((input_w - pooling_w) / stride_w_ + 1));

    std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
    if (output_data == nullptr || output_data->empty()) {
      output_data =
          std::make_shared<Tensor<float>>(input_c, output_h, output_w);
      outputs.at(i) = output_data;
    }

    CHECK(output_data->rows() == output_h && output_data->cols() == output_w &&
          output_data->channels() == input_c)
        << "The output size of max pooling layer is error";

    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat& input_channel = input_data_->at(ic);
      arma::fmat& output_channel = output_data->at(ic);
      for (uint32_t c = 0; c < input_w - pooling_w + 1; c += stride_w_) {
        for (uint32_t r = 0; r < input_h - pooling_h + 1; r += stride_h_) {
          float* output_channel_ptr = output_channel.colptr(int(c / stride_w_));
          float max_value = std::numeric_limits<float>::lowest();
          for (uint32_t w = 0; w < pooling_w; ++w) {
            const float* col_ptr = input_channel.colptr(c + w) + r;
            for (uint32_t h = 0; h < pooling_h; ++h) {
              float current_value = *(col_ptr + h);
              max_value = max_value > current_value ? max_value : current_value;
            }
          }
          *(output_channel_ptr + int(r / stride_h_)) = max_value;
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