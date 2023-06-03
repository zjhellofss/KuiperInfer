//
// Created by fss on 22-11-13.
//

#include <float.h>
#include <glog/logging.h>
#include <numeric>
#include "layer/abstract/layer_factory.hpp"
#include "softmax.hpp"
#include "utils/gpu_utils.cuh"


namespace kuiper_infer {

__global__ void kernel_channel_max(const int num, const int channels,
                                   const int spatial_dim, const float* data,
                                   float* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

__global__ void kernel_channel_subtract(const int count, const int num,
                                        const int channels,
                                        const int spatial_dim,
                                        const float* channel_max, float* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

__global__ void kernel_exp(const int count, const float* data, float* out) {
  CUDA_KERNEL_LOOP(index, count) { out[index] = exp(data[index]); }
}

__global__ void kernel_channel_sum(const int num, const int channels,
                                   const int spatial_dim, const float* data,
                                   float* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

__global__ void kernel_channel_div(const int count, const int num,
                                   const int channels, const int spatial_dim,
                                   const float* channel_sum, float* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

__global__ void kernel_channel_dot(const int num, const int channels,
                                   const int spatial_dim, const float* data_1,
                                   const float* data_2, float* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s] *
              data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

SoftmaxLayer::SoftmaxLayer(int dim) : Layer("Softmax"), softmax_dim_(dim) {}

InferStatus SoftmaxLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the softmax layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the softmax layer "
                  "do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const auto& input = inputs[0];

//   auto top_data = outputs[0]->mutable_data<float>();

  uint32_t count = inputs.at(0)->size();

  
  uint32_t channels = inputs.at(0)->nums();
  uint32_t outer_num_ = inputs.at(0)->nums();
  uint32_t inner_num_ =inputs.at(0)->rows();
  

  auto top_data = outputs[0]->gpu_data();
  auto bottom_data=inputs[0]->gpu_data();

  

  auto scale_data=std::make_shared<Tensor<float>>(std::vector<uint32_t>{outer_num_ * inner_num_}).get()->gpu_data();


  kernel_channel_max<<<KUIPER_GET_BLOCKS(outer_num_ * inner_num_),
                       KUIPER_CUDA_NUM_THREADS>>>(
      outer_num_, channels, inner_num_, top_data, scale_data);
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<<<KUIPER_GET_BLOCKS(count),
                            KUIPER_CUDA_NUM_THREADS>>>(
      count, outer_num_, channels, inner_num_, scale_data, top_data);
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
      count, top_data, top_data);
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<<<KUIPER_GET_BLOCKS(outer_num_ * inner_num_),
                       KUIPER_CUDA_NUM_THREADS>>>(
      outer_num_, channels, inner_num_, top_data, scale_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
      count, outer_num_, channels, inner_num_, scale_data, top_data);
  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus SoftmaxLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& softmax_layer) {
  CHECK(op != nullptr) << "SoftMax operator is nullptr";
  const auto& params = op->params;
  if (params.find("dim") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingDim;
  }
  auto dim_param = params.at("dim");
  auto dim = dynamic_cast<RuntimeParameterInt*>(dim_param);
  if (dim == nullptr) {
    return ParseParameterAttrStatus::kParameterMissingDim;
  }
  softmax_layer = std::make_shared<SoftmaxLayer>(dim->value);  // 创建softmax层
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}
LayerRegistererWrapper kSoftMaxGetInstanceNN("nn.Softmax",
                                             SoftmaxLayer::GetInstance);
LayerRegistererWrapper kSoftMaxGetInstanceF("F.softmax",
                                            SoftmaxLayer::GetInstance);
}  // namespace kuiper_infer