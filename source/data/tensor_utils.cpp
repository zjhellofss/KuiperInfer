//
// Created by fss on 2023/3/20.
//
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer {
bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                  const std::shared_ptr<Tensor<float>>& b, float threshold) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  if (a->shapes() != b->shapes()) {
    return false;
  }
  bool is_same = arma::approx_equal(a->data(), b->data(), "absdiff", threshold);
  return is_same;
}

void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    CHECK(tensor1->shapes() == output_tensor->shapes());
    output_tensor->set_data(tensor1->data() + tensor2->data());
  } else {
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
          output_tensor->shapes() == input_tensor2->shapes());
    output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
  }
}

void TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2,
    const std::shared_ptr<Tensor<float>>& output_tensor) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    CHECK(tensor1->shapes() == output_tensor->shapes());
    output_tensor->set_data(tensor1->data() % tensor2->data());
  } else {
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
          output_tensor->shapes() == input_tensor2->shapes());
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
  }
}

std::shared_ptr<Tensor<float>> TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    sftensor output_tensor = TensorCreate(tensor1->shapes());
    output_tensor->set_data(tensor1->data() + tensor2->data());
    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    sftensor output_tensor = TensorCreate(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
    return output_tensor;
  }
}

std::shared_ptr<Tensor<float>> TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    sftensor output_tensor = TensorCreate(tensor1->shapes());
    output_tensor->set_data(tensor1->data() % tensor2->data());
    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    sftensor output_tensor = TensorCreate(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
    return output_tensor;
  }
}

std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols) {
  return std::make_shared<Tensor<float>>(channels, rows, cols);
}

std::shared_ptr<Tensor<float>> TensorCreate(
    const std::vector<uint32_t>& shapes) {
  CHECK(shapes.size() == 3);
  return TensorCreate(shapes.at(0), shapes.at(1), shapes.at(2));
}

std::shared_ptr<Tensor<float>> TensorPadding(
    const std::shared_ptr<Tensor<float>>& tensor,
    const std::vector<uint32_t>& pads, float padding_value) {
  CHECK(tensor != nullptr && !tensor->empty());
  CHECK(pads.size() == 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  std::shared_ptr<ftensor> output = std::make_shared<ftensor>(
      tensor->channels(), tensor->rows() + pad_rows1 + pad_rows2,
      tensor->cols() + pad_cols1 + pad_cols2);

  const uint32_t channels = tensor->channels();
  for (uint32_t channel = 0; channel < channels; ++channel) {
    const arma::fmat& in_channel = tensor->slice(channel);
    arma::fmat& output_channel = output->slice(channel);
    const uint32_t in_channel_width = in_channel.n_cols;
    const uint32_t in_channel_height = in_channel.n_rows;

    for (uint32_t w = 0; w < in_channel_width; ++w) {
      float* output_channel_ptr =
          const_cast<float*>(output_channel.colptr(w + pad_cols1));
      const float* in_channel_ptr = in_channel.colptr(w);
      for (uint32_t h = 0; h < in_channel_height; ++h) {
        const float value = *(in_channel_ptr + h);
        *(output_channel_ptr + h + pad_rows1) = value;
      }

      for (uint32_t h = 0; h < pad_rows1; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }

      for (uint32_t h = 0; h < pad_rows2; ++h) {
        *(output_channel_ptr + in_channel_height + pad_rows1 + h) =
            padding_value;
      }
    }

    for (uint32_t w = 0; w < pad_cols1; ++w) {
      float* output_channel_ptr = const_cast<float*>(output_channel.colptr(w));
      for (uint32_t h = 0; h < in_channel_height + pad_rows1 + pad_rows2; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }
    }

    for (uint32_t w = 0; w < pad_cols2; ++w) {
      float* output_channel_ptr = const_cast<float*>(
          output_channel.colptr(pad_cols1 + w + in_channel_width));
      for (uint32_t h = 0; h < in_channel_height + pad_rows1 + pad_rows2; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }
    }
  }
  return output;
}

std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1,
                                               const sftensor& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    return {tensor1, tensor2};
  } else {
    CHECK(tensor1->channels() == tensor2->channels());
    if (tensor2->rows() == 1 && tensor2->cols() == 1) {
      sftensor new_tensor =
          TensorCreate(tensor2->channels(), tensor1->rows(), tensor1->cols());
      CHECK(tensor2->size() == tensor2->channels());
      for (uint32_t c = 0; c < tensor2->channels(); ++c) {
        new_tensor->slice(c).fill(tensor2->index(c));
      }
      return {tensor1, new_tensor};
    } else if (tensor1->rows() == 1 && tensor1->cols() == 1) {
      sftensor new_tensor =
          TensorCreate(tensor1->channels(), tensor2->rows(), tensor2->cols());
      CHECK(tensor1->size() == tensor1->channels());
      for (uint32_t c = 0; c < tensor1->channels(); ++c) {
        new_tensor->slice(c).fill(tensor1->index(c));
      }
      return {new_tensor, tensor2};
    } else {
      LOG(FATAL) << "Broadcast shape is not adapting!";
      return {tensor1, tensor2};
    }
  }
}

std::shared_ptr<Tensor<float>> TensorClone(
    std::shared_ptr<Tensor<float>> tensor) {
  return std::make_shared<Tensor<float>>(*tensor);
}
}  // namespace kuiper_infer