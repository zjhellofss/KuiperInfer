#include "cat.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "utils/gpu_utils.cuh"

namespace kuiper_infer {

__global__ void Concat(const int nthreads, const float* in_data,
                       const int concat_size, const int top_concat_axis,
                       const int bottom_concat_axis,
                       const int offset_concat_axis, float* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // bottom_concat_axis 指的是当前需要合并的维度
    // concat_size 指的是当前合并的维度下一维的矩阵的大小
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index =
        concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;

    out_data[top_index] = in_data[index];
  }
}

CatLayer::CatLayer(int dim) : Layer("cat"), dim_(dim) {}

InferStatus CatLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty() || inputs.size() < 2) {
    LOG(ERROR) << "The input tensor array in the cat layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (dim_ != 1 && dim_ != -3) {
    LOG(ERROR) << "The dimension parameter of cat layer is error";
    return InferStatus::kInferFailedDimensionParameterError;
  }

  auto past_shape = inputs[0]->shapes();

  for (int i = 1; i < inputs.size(); i++) {
    auto input_item = inputs[i];
    for (int j = 0; j < past_shape.size(); j++) {
      if (j == dim_) {
        continue;
      }
      if (past_shape[j] != input_item->shapes()[j]) {
        LOG(ERROR) << "The input tensor shape is not match";
        return InferStatus::kInferFailedInputEmpty;
      }
    }
    past_shape = input_item->shapes();
  }


  // 计算需要合并后，合并维度维度的大小
  uint32_t concat_axis_size = 0;

  for (auto tensor : inputs) {
    concat_axis_size += tensor->shapes()[dim_];
  }

  // 合并的下一维的矩阵的大小
  uint32_t concat_input_size = 1;

  for (int i = dim_ + 1; i < inputs[0]->shapes().size(); i++) {
    concat_input_size *=inputs[0]->shapes()[i];
  }

  if (outputs.size() == 0) {
    auto output_shape = inputs.at(0)->shapes();
    output_shape[dim_] = concat_axis_size;

    std::shared_ptr<Tensor<float>> output =
        std::make_shared<Tensor<float>>(output_shape);
    outputs.push_back(output);
  }


  float* top_data = outputs.at(0)->gpu_data();
  uint32_t offset_concat_axis = 0;

  for (int i = 0; i < inputs.size(); ++i) {
    const float* bottom_data = inputs[i]->gpu_data();
    const int bottom_concat_axis = inputs[i]->shapes()[dim_];
    const int bottom_concat_size = bottom_concat_axis * concat_input_size;
    const int count = inputs[i]->size();
    Concat<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
        count, bottom_data, concat_input_size, concat_axis_size,
        bottom_concat_axis, offset_concat_axis, top_data);
    offset_concat_axis += bottom_concat_axis;
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus CatLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& cat_layer) {
  CHECK(op != nullptr) << "Cat operator is nullptr";
  const auto& params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";
  if (params.find("dim") == params.end()) {
    LOG(ERROR) << "Can not find the dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  const auto& dim_param = dynamic_cast<RuntimeParameterInt*>(params.at("dim"));
  if (!dim_param) {
    LOG(ERROR) << "Can not find the dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }
  const int32_t dim = dim_param->value;
  cat_layer = std::make_shared<CatLayer>(dim);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kCatGetInstance("torch.cat", CatLayer::GetInstance);
}  // namespace kuiper_infer