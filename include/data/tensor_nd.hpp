//
// Created by fss on 23-5-25.
//

#ifndef KUIPER_INFER_INCLUDE_DATA_TENSOR_ND_HPP_
#define KUIPER_INFER_INCLUDE_DATA_TENSOR_ND_HPP_

#include <glog/logging.h>
#include <algorithm>
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>

template <uint32_t HeadShape, uint32_t... TailShapes>
static void StoreShape(std::vector<uint32_t>& shapes) {
  shapes.push_back(HeadShape);
  if constexpr (sizeof...(TailShapes) > 0) {
    StoreShape<TailShapes...>(shapes);
  }
}

template <typename T = float, uint32_t... Shapes>
class TensorNd;

// 四维以上的张量
template <uint32_t... Shapes>
class TensorNd<float, Shapes...> {
 public:
  explicit TensorNd() {
    if constexpr (sizeof...(Shapes) > 0) {
      StoreShape<Shapes...>(this->raw_shapes_);
    }
    if (!raw_shapes_.empty()) {
      const uint32_t size = std::accumulate(
          raw_shapes_.begin(), raw_shapes_.end(), 1, std::multiplies());
      float* data_ptr = new float[size];
      data_ = std::shared_ptr<float>(
          data_ptr, [](const float* data_ptr) { delete[] data_ptr; });
    }
  }

  explicit TensorNd(const std::vector<uint32_t>& shapes) {
    static_assert(sizeof...(Shapes) == 0);
    this->raw_shapes_ = shapes;
    const uint32_t size = std::accumulate(
        raw_shapes_.begin(), raw_shapes_.end(), 1, std::multiplies());
    float* data_ptr = new float[size];
    data_ = std::shared_ptr<float>(
        data_ptr, [](const float* data_ptr) { delete[] data_ptr; });
  }

  const std::vector<uint32_t>& raw_shapes() const { return this->raw_shapes_; }

  float* raw_ptr() const {
    CHECK(this->data_ != nullptr);
    float* data_ptr = this->data_.get();
    return data_ptr;
  }

  float* raw_ptr(uint32_t offset) const {
    float* data_ptr = this->data_.get();
    CHECK(data_ptr != nullptr);
    CHECK(offset < this->size());
    return data_ptr + offset;
  }

  float* raw_ptr(const std::vector<uint32_t>& offsets) const {
    CHECK_LE(offsets.size(), this->raw_shapes_.size());
    std::vector<uint32_t> offsets_;
    for (int i = 0; i < raw_shapes_.size(); ++i) {
      if (i < offsets.size()) {
        offsets_.push_back(offsets.at(i));
      } else {
        offsets_.push_back(0);
      }
    }

    CHECK_EQ(offsets_.size(), this->raw_shapes_.size());
    uint32_t total_offset = 0;
    for (int i = 0; i < offsets_.size(); ++i) {
      total_offset *= raw_shapes_.at(i);
      CHECK_GE(offsets_[i], 0);
      CHECK_LT(offsets_[i], raw_shapes_.at(i));
      total_offset += offsets_[i];
    }
    return this->raw_ptr(total_offset);
  }

  float* slice_ptr(uint32_t slice_index) {
    std::vector<uint32_t> offsets{slice_index};
    return this->raw_ptr(offsets);
  }

  bool empty() const {
    return this->data_ == nullptr || this->raw_shapes_.empty();
  }

  void Fill(float value) {
    CHECK_NE(this->data_, nullptr);
    CHECK_GT(this->raw_shapes_.size(), 0);
    const uint32_t size = this->size();
    float* data = this->raw_ptr();
    for (uint32_t i = 0; i < size; ++i) {
      *(data + i) = value;
    }
  }

  float index(uint32_t offset) const {
    float* raw_ptr = this->raw_ptr(offset);
    CHECK_NE(raw_ptr, nullptr);
    return *raw_ptr;
  }

  float& index(uint32_t offset) {
    float* raw_ptr = this->raw_ptr(offset);
    CHECK_NE(raw_ptr, nullptr);
    return *raw_ptr;
  }

  void Fill(const std::vector<float>& values) {
    CHECK_NE(this->data_, nullptr);
    CHECK_GT(this->raw_shapes_.size(), 0);
    const uint32_t size = this->size();
    const uint32_t value_size = values.size();
    CHECK_EQ(size, value_size);
    std::copy(values.begin(), values.end(), this->data_.get());
  }

  void Fill(const float* ptr, uint32_t size) {
    CHECK_NE(this->data_, nullptr);
    CHECK_EQ(size, this->size());
    float* data_ptr = this->data_.get();
    CHECK_NE(data_ptr, nullptr);
    for (uint32_t i = 0; i < size; ++i) {
      *(data_ptr + i) = *(ptr + i);
    }
  }

  uint32_t size() const {
    CHECK_GT(this->raw_shapes().size(), 0);
    const uint32_t size = std::accumulate(
        raw_shapes_.begin(), raw_shapes_.end(), 1, std::multiplies());
    return size;
  }

  uint32_t channels() const {
    CHECK_GE(raw_shapes_.size(), 1);
    return raw_shapes_.at(0);
  }

  uint32_t rows() const {
    CHECK_GE(raw_shapes_.size(), 2);
    return raw_shapes_.at(1);
  }

  uint32_t cols() const {
    CHECK_GE(raw_shapes_.size(), 3);
    return raw_shapes_.at(2);
  }

 private:
  std::vector<uint32_t> raw_shapes_;
  std::shared_ptr<float> data_ = nullptr;
};

template <uint32_t... Shapes>
using ftensor_nd = TensorNd<float, Shapes...>;

template <uint32_t... Shapes>
using sftensor_nd = std::shared_ptr<TensorNd<float, Shapes...>>;

using ftensor_nd0 = TensorNd<float>;

#endif  // KUIPER_INFER_INCLUDE_DATA_TENSOR_ND_HPP_
