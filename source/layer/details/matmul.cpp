// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
// Created by fss on 24-2-9.
//
#include "matmul.hpp"
namespace kuiper_infer {
LLamaMatmulLayer::LLamaMatmulLayer(int32_t weight_dim0, int32_t weight_dim1)
    : ParamLayer("matmul"), weight_dim0_(weight_dim0), weight_dim1_(weight_dim1) {
  this->weights_.resize(1);
}

void LLamaMatmulLayer::set_weights(const std::vector<float>& weights) {
  LOG(FATAL) << "This function is not implement in the LLamaMatmul layer！";
}

void LLamaMatmulLayer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  CHECK(weights.size() == weights_.size());
  this->weights_ = weights;
}

StatusCode LLamaMatmulLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the matmul layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the matmul layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the matmul "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  if (this->weights_.empty()) {
    LOG(ERROR) << "The weight tensor in the matmul layer is empty";
    return StatusCode::kInferParameterError;
  }

  if (weights_.size() != 1) {
    LOG(ERROR) << "Need one weight tensor in the matmul layer";
    return StatusCode::kInferParameterError;
  }

  // w @ x
  uint32_t batch = inputs.size();
#pragma omp parallel for if (batch > 1) num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    std::shared_ptr<Tensor<float>> input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the matmul layer has an empty tensor " << i << " th";
    const std::vector<uint32_t>& input_shapes = input->raw_shapes();
    CHECK(!input_shapes.empty() && input_shapes.size() <= 2);

    uint32_t input_dim0 = 1;
    uint32_t input_dim1 = 1;
    if (input_shapes.size() == 1) {
      input_dim0 = input_shapes.front();
    } else {
      input_dim0 = input_shapes.at(0);
      input_dim1 = input_shapes.at(1);
    }
    CHECK_EQ(input_dim0, weight_dim1_);

    // xt
    arma::fmat input_vec(input->raw_ptr(), input_dim1, input_dim0, false, true);
    const std::shared_ptr<Tensor<float>>& weight = weights_.front();
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(1, input_dim1, weight_dim0_);
      outputs.at(i) = output;
    }

    const auto& output_raw_shapes = output->shapes();
    if (output_raw_shapes.size() == 3) {
      CHECK(output_raw_shapes.at(1) == weight_dim0_ && output_raw_shapes.at(2) == input_dim1)
          << "The row of output tensor should be same to input dim 1 and the "
             "col of output tensor should be same to weight dim 0.";
    } else {
      LOG(FATAL) << "The shape of output tensor need be equal to one or two";
    }
    if (input_dim1 == 1) {
      float* output_ptr = output->raw_ptr();
      float* weight_ptr = weight->raw_ptr();
#pragma omp parallel for
      for (int32_t j = 0; j < weight_dim0_; ++j) {
        arma::fmat sub_weight(weight_ptr + j * weight_dim1_, weight_dim1_, 1, false, true);
        *(output_ptr + j) = arma::as_scalar(input_vec * sub_weight);
      }
    } else {
      arma::fmat weight_data(weight->raw_ptr(), weight_dim1_, weight_dim0_, false, true);  // wt
      if (weight_dim0_ == 1) {
        arma::fmat output_mat(output->raw_ptr(), input_dim1, weight_dim0_, false, true);
        output_mat = input_vec * weight_data;
      } else {
        arma::fmat output_mat(output->raw_ptr(), weight_dim0_, input_dim1, false, true);
        output_mat = (input_vec * weight_data).t();
      }
    }
  }
  return StatusCode::kSuccess;
}
}  // namespace kuiper_infer