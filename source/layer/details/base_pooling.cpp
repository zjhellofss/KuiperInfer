//
// Created by fss on 23-10-20.
//
#include "base_pooling.hpp"
namespace kuiper_infer {
void BasePoolingLayer::ComputeOutput(const sftensor input_data, sftensor output_data, uint32_t input_h,
                                     uint32_t input_w, uint32_t input_c, uint32_t pooling_h,
                                     uint32_t pooling_w, uint32_t padding_h, uint32_t padding_w,
                                     uint32_t stride_h, uint32_t stride_w,
                                     kuiper_infer::PoolingType pooling_type) const {
  CHECK_NE(input_data, nullptr) << "The input tensor of the pooling operator is nullptr";
  CHECK_NE(output_data, nullptr) << "The output tensor of the pooling operator is nullptr";
  const uint32_t input_padded_h = input_data->rows() + 2 * padding_h;
  const uint32_t input_padded_w = input_data->cols() + 2 * padding_w;

  for (uint32_t ic = 0; ic < input_c; ++ic) {
    const arma::fmat& input_channel = input_data->slice(ic);
    arma::fmat& output_channel = output_data->slice(ic);
    for (uint32_t c = 0; c < input_padded_w - pooling_w + 1; c += stride_w) {
      uint32_t output_col = uint32_t(c / stride_w);
      for (uint32_t r = 0; r < input_padded_h - pooling_h + 1; r += stride_h) {
        std::vector<float> window_values;
        uint32_t output_row = uint32_t(r / stride_h);
        float* output_channel_ptr = output_channel.colptr(output_col);

        float current_value = 0.f;
        for (uint32_t w = 0; w < pooling_w; ++w) {
          const float* col_ptr = input_channel.colptr(c + w - padding_w) + r;
          for (uint32_t h = 0; h < pooling_h; ++h) {
            if ((h + r >= padding_h && w + c >= padding_w) &&
                (h + r < input_h + padding_h && w + c < input_w + padding_w)) {
              current_value = *(col_ptr + h - padding_h);
            } else {
              if (pooling_type == PoolingType::kMaxPooling) {
                current_value = std::numeric_limits<float>::lowest();
              } else {
                current_value = 0.f;
              }
            }
            window_values.push_back(current_value);
          }
        }
        *(output_channel_ptr + output_row) = (pooling_type == PoolingType::kMaxPooling)
                                                 ? ReduceMax(window_values)
                                                 : ReduceAverage(window_values);
      }
    }
  }
}

float BasePoolingLayer::ReduceMax(const std::vector<float>& window_values) const {
  CHECK_EQ(window_values.empty(), false);
  float max_value = *std::max_element(window_values.begin(), window_values.end());
  return max_value;
}

float BasePoolingLayer::ReduceAverage(const std::vector<float>& window_values) const {
  CHECK_EQ(window_values.empty(), false);
  float average_value = std::accumulate(window_values.begin(), window_values.end(), 0.f);
  return average_value / (float)window_values.size();
}
}  // namespace kuiper_infer