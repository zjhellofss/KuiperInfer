//
// Created by fss on 22-11-12.
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include "utils/gpu_utils.cuh"

namespace kuiper_infer {

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  this->size_ = channels * rows * cols;

  cudaMalloc((void**)&this->gpu_data_, sizeof(float) * this->size_);

  this->shapes_ = std::vector<uint32_t>{1, channels, rows, cols};
}

Tensor<float>::Tensor(uint32_t nums, uint32_t channels, uint32_t rows,
                      uint32_t cols) {
  this->size_ = nums * channels * rows * cols;
  cudaMalloc((void**)&this->gpu_data_, sizeof(float) * this->size_);

  this->shapes_ = std::vector<uint32_t>{nums, channels, rows, cols};
}

Tensor<float>::Tensor(const Tensor& tensor) {
  if (this != &tensor) {
    cudaMemcpy(this->gpu_data_, tensor.gpu_data_, tensor.size_ * sizeof(float),
               cudaMemcpyDeviceToDevice);
    this->shapes_ = tensor.shapes_;
  }
}

Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
  CHECK(shapes.size() > 0);
  switch (shapes.size()) {
    case 1:
      this->shapes_ = std::vector<uint32_t>{1, 1, 1, shapes[0]};
      break;
    case 2:
      this->shapes_ = std::vector<uint32_t>{1, 1, shapes[0], shapes[1]};
      break;
    case 3:
      this->shapes_ = std::vector<uint32_t>{1, shapes[0], shapes[1], shapes[2]};
      break;
    case 4:
      this->shapes_ = shapes;
    default:
      break;
  }
  this->size_ =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<int>());
  cudaMalloc((void**)&this->gpu_data_, sizeof(float) * this->size_);
}

Tensor<float>::~Tensor() { cudaFree(this->gpu_data_); }

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if (this != &tensor) {
    cudaMemcpy(this->gpu_data_, tensor.gpu_data_, tensor.size_ * sizeof(float),
               cudaMemcpyDeviceToDevice);
    this->shapes_ = tensor.shapes_;
  }
  return *this;
}
uint32_t Tensor<float>::nums() const {
  CHECK(!this->empty());
  return this->shapes_[0];
}

uint32_t Tensor<float>::channels() const {
  //  CHECK(!this->gpu_data_.empty());
  return this->shapes_[1];
}

uint32_t Tensor<float>::rows() const {
  //  CHECK(!this->gpu_data_.empty());
  return this->shapes_[2];
}

uint32_t Tensor<float>::cols() const {
  //  CHECK(!this->gpu_data_.empty());
  return this->shapes_[3];
}
bool Tensor<float>::empty() const { return this->gpu_data_ == nullptr; }

float Tensor<float>::index(uint32_t nums, uint32_t channels, uint32_t rows,
                           uint32_t cols) {
  uint32_t offset = nums * shapes_[1] * shapes_[2] * shapes_[3] +
                    channels * shapes_[2] * shapes_[3] + rows * shapes_[3] +
                    cols;

  CHECK(offset < this->size_) << "Tensor index out of bound!";

  return this->gpu_data_[offset];
}

std::vector<uint32_t> Tensor<float>::shapes() {
  //  CHECK(!this->gpu_data_.empty());
  return this->shapes_;
}

/**
 * 初始化GPU中的数据
 */
void Tensor<float>::Rand() {
  CHECK(!this->empty());

  rand_array(this->size_, this->gpu_data_);
}

float* Tensor<float>::data() { return this->gpu_data_; }

void Tensor<float>::Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
  //  CHECK(!this->gpu_data_.empty());
  //  CHECK_EQ(pads.size(), 4);
  //  uint32_t pad_rows1 = pads.at(0);  // up
  //  uint32_t pad_rows2 = pads.at(1);  // bottom
  //  uint32_t pad_cols1 = pads.at(2);  // left
  //  uint32_t pad_cols2 = pads.at(3);  // right
  //
  //  arma::fcube new_data(this->gpu_data_.n_rows + pad_rows1 + pad_rows2,
  //                       this->gpu_data_.n_cols + pad_cols1 + pad_cols2,
  //                       this->gpu_data_.n_slices);
  //  new_data.fill(padding_value);
  //
  //  new_data.subcube(pad_rows1, pad_cols1, 0, new_data.n_rows - pad_rows2 - 1,
  //                   new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1) =
  //      this->gpu_data_;
  //  this->gpu_data_ = std::move(new_data);
}

void Tensor<float>::Fill(float value) {
  CHECK(!this->empty());

  element_wise_fill(this->size_, this->gpu_data_, value);
}

void Tensor<float>::Fill(const std::vector<float>& values) {
  CHECK(!this->empty());
  const uint32_t total_elems = this->size();
  CHECK_EQ(values.size(), total_elems);

  const uint32_t rows = this->rows();
  const uint32_t cols = this->cols();
  const uint32_t planes = rows * cols;
  const uint32_t channels = this->channels();

  // 在CUDA设备上分配内存
  cudaMalloc((void**)&this->gpu_data_, values.size() * sizeof(float));
  // 将数据从主机复制到设备
  cudaMemcpy(this->gpu_data_, values.data(), values.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  // for (auto item : values) {
  //   std::cout << item << " " << std::endl;
  // }
}

void Tensor<float>::Show() {
  //  for (uint32_t i = 0; i < this->channels(); ++i) {
  //    LOG(INFO) << "Channel: " << i;
  //    LOG(INFO) << "\n" << this->gpu_data_.slice(i);
  //  }void
}

// void Tensor<float>::Flatten(bool row_major) {
//   CHECK(!this->empty());
//   const uint32_t size = this->gpu_data_.size();
//   this->Reshape({size}, row_major);
// }

// void Tensor<float>::Rand() {
//   CHECK(!this->empty());
//   this->gpu_data_.randn();
// }

void Tensor<float>::Ones() {
  CHECK(!this->empty());
  this->Fill(1.f);
}

// void Tensor<float>::Transform(const std::function<float(float)>& filter) {
//   CHECK(!this->gpu_data_.empty());
//   this->gpu_data_.transform(filter);
// }

const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
  CHECK(!this->shapes_.empty());
  CHECK_LE(this->shapes_.size(), 3);
  CHECK_GE(this->shapes_.size(), 1);
  return this->shapes_;
}

void Tensor<float>::slice_fill(uint32_t channel, float val) {
  // CHECK_LT(channel, this->channels());
  // return this->data_.slice(channel);
}

void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes,
                            bool row_major) {
  CHECK(!this->empty());
  CHECK(!shapes.empty());
  const uint32_t origin_size = this->size();
  const uint32_t current_size = std::accumulate(shapes.begin(), shapes.end(), 1,
                                                std::multiplies<uint32_t>());
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);

  this->shapes_ = shapes;
}

float* Tensor<float>::gpu_data() { return this->gpu_data_; }

uint32_t Tensor<float>::size() { return this->size_; }

//
std::shared_ptr<float> Tensor<float>::to_cpu_data() {
  CHECK(!this->empty());

  float* data = new float[this->size_];
  cudaMemcpy(data, this->gpu_data_, this->size_ * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::shared_ptr<float> cpu_data_ptr(data, [](float* ptr) {
    delete[] ptr;  // 使用delete[]释放数组内存
  });

  return cpu_data_ptr;
}



}  // namespace kuiper_infer
