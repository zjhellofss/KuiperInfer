

#include <glog/logging.h>
#include "data/tensor_util.hpp"
#include "utils/gpu_utils.cuh"



namespace kuiper_infer {









std::shared_ptr<Tensor<float>> TensorCreate(uint32_t nums,uint32_t channels, uint32_t rows,
                                            uint32_t cols) {
  return std::make_shared<Tensor<float>>(nums,channels, rows, cols);
}

std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols) {
  return std::make_shared<Tensor<float>>(1,channels, rows, cols);
}

std::shared_ptr<Tensor<float>> TensorCreate(std::vector<uint32_t>& shapes) {
  return TensorCreate(shapes);
}

std::shared_ptr<Tensor<float>> TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {



    std::shared_ptr<Tensor<float>> output_tensor=std::make_shared<Tensor<float>>(tensor1.get()->shapes());

    int count = tensor1->size();

    float* inputs_data1 = tensor1->data();
    float* inputs_data2 = tensor2->data();
    float* outputs_data = output_tensor->data();
    // element_add<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
    //     count, inputs_data1, inputs_data2, outputs_data);

    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    // const auto& [input_tensor1, input_tensor2] =
    //     TensorBroadcast(tensor1, tensor2);
    // CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    // std::shared_ptr<Tensor<float>> output_tensor =
    //     TensorCreate(input_tensor1->shapes());
    // int count = tensor1->size();

    // auto inputs_data1 = tensor1->data();
    // auto inputs_data2 = tensor2->data();
    // auto outputs_data = output_tensor->data();
    // element_add<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
    //     count, inputs_data1, input_data2, outputs_data);
    return nullptr;
  }
}

void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    CHECK(tensor1->shapes() == output_tensor->shapes());

    int count = tensor1->size();

    auto inputs_data1 = tensor1->data();
    auto inputs_data2 = tensor2->data();
    auto outputs_data = output_tensor->data();
    // element_add<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
    //     count, inputs_data1, inputs_data2, outputs_data);

  } else {
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    // const auto& [input_tensor1, input_tensor2] =
    //     TensorBroadcast(tensor1, tensor2);
    // CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
    //       output_tensor->shapes() == input_tensor2->shapes());
    // // output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
  }
}

std::tuple<std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>>
TensorBroadcast(const std::shared_ptr<Tensor<float>>& tensor1,
                const std::shared_ptr<Tensor<float>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    return {tensor1, tensor2};
  } else {
    CHECK(tensor1->channels() == tensor2->channels());
    if (tensor2->rows() == 1 && tensor2->cols() == 1) {
      std::shared_ptr<Tensor<float>> new_tensor =
          TensorCreate(tensor2->channels(), tensor1->rows(), tensor1->cols());
      CHECK(tensor2->size() == tensor2->channels());

      for (uint32_t c = 0; c < tensor2->channels(); ++c) {
        new_tensor->slice_fill(c, tensor2->index(1, c, 1, 1));
      }
      return {tensor1, new_tensor};

    } else if (tensor1->rows() == 1 && tensor1->cols() == 1) {
      std::shared_ptr<Tensor<float>> new_tensor =
          TensorCreate(tensor1->channels(), tensor2->rows(), tensor2->cols());
      CHECK(tensor1->size() == tensor1->channels());
      for (uint32_t c = 0; c < tensor1->channels(); ++c) {
        new_tensor->slice_fill(c, tensor1->index(1, c, 1, 1));
      }
      return {new_tensor, tensor2};
    } else {
      LOG(FATAL) << "Broadcast shape is not adapting!";
      return {tensor1, tensor2};
    }
  }
}

}  // namespace kuiper_infer