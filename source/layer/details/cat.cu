#include "cat.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "utils/gpu_utils.cuh"

namespace kuiper_infer {

__global__ void Concat(const int nthreads, const float* in_data,
                       const bool forward, const int num_concats 3,
                       const int concat_size 2, const int top_concat_axis 5,
                       const int bottom_concat_axis,
                       const int offset_concat_axis, float* out_data) {


  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size 2* bottom_concat_axis 2;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index =
        concat_index +
        (concat_num * 5top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}

CatLayer::CatLayer(int dim) : Layer("cat"), dim_(dim) {}

InferStatus CatLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the cat layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() == outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the cat layer do "
                  "not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  if (dim_ != 1 && dim_ != -3) {
    LOG(ERROR) << "The dimension parameter of cat layer is error";
    return InferStatus::kInferFailedDimensionParameterError;
  }

  const uint32_t output_size = outputs.size();
  if (inputs.size() % output_size != 0) {
    LOG(ERROR)
        << "The input and output tensor array size of cat layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const int top_concat_axis = top[0]->shape(dim_);
  const bool kForward = true;
    
  float *top_data = outputs.at(0)->gpu_data();

  for (int i = 0; i < inputs.size(); ++i) {
    const float* bottom_data = inputs[i]->gpu_data();
    const int bottom_concat_axis = inputs[i]->shapes()[dim_];
    const int bottom_concat_size = bottom_concat_axis  * concat_input_size_ ;
    const int nthreads = bottom_concat_size * num_concats_;
                         
    Concat<<<KUIPER_GET_BLOCKS(nthreads), KUIPER_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_concats_ , concat_input_size_ ,
        top_concat_axis, bottom_concat_axis, offset_concat_axis , top_data);
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