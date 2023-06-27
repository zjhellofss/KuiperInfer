#include <float.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/adaptive_avgpooling.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"
using namespace std;
using namespace kuiper_infer;

std::shared_ptr<float> AdaptiveAveragePooling(std::shared_ptr<float> inputs, const int nums,
                                  const int channels, const int input_h,
                                  const int input_w, const int output_h,
                                  const int output_w, const int kernel_h,
                                  const int kernel_w, const int stride_h,
                                  const int stride_w) {
  float* output = new float[(nums * channels * output_h * output_w)];

  for (int n = 0; n < nums; n++) {
    for (int c = 0; c < channels; c++) {
      for (int ph = 0; ph < output_h; ph++) {
        for (int pw = 0; pw < output_w; pw++) {
          int hstart = ph * stride_h;
          int wstart = pw * stride_w;
          const int hend = hstart + kernel_h;
          const int wend = wstart + kernel_w;
          float avgval = 0;

          const float* const input_slice =
              inputs.get() + (n * channels + c) * input_h * input_w;

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              avgval += input_slice[h * input_w + w];
            }
          }
          const uint32_t kernel_size = kernel_h * kernel_w;
          output[n * channels * output_h * output_w + c * output_h * output_w +
                 ph * output_w + pw] = avgval / kernel_size;
        }
      }
    }
  }
  std::shared_ptr<float> outputs(output, [](float* ptr) {
    delete[] ptr;  // 使用delete[]释放数组内存
  });
  return outputs;
}

TEST(test_layer, forward_adaptive_average_pooling) {

  uint32_t nums = 3;
  uint32_t channels = 3;
  uint32_t row = 244;
  uint32_t col = 244;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(nums, channels, row, col);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  input->Rand();

  inputs.push_back(input);

  const uint32_t output_h = 242;
  const uint32_t output_w = 242;

  const uint32_t stride_h = uint32_t(std::floor(row / output_h));
  const uint32_t stride_w = uint32_t(std::floor(col / output_w));

  const uint32_t kernel_h = 
        (int)row - (int(output_h) - 1) * int(stride_h);
  const uint32_t kernel_w = 
        (int)col - (int(output_w) - 1) - int(stride_w);

  

  AdaptiveAveragePoolingLayer adaptive_avg_pool_layer(output_h, output_w);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  adaptive_avg_pool_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 1);
  auto right_answer =
      AdaptiveAveragePooling(input->to_cpu_data(), nums, channels, row, col, output_h,
                 output_w, kernel_h, kernel_w, stride_h, stride_w);

  auto layer_output = outputs.at(0)->to_cpu_data();
  for (int i = 0; i < outputs.at(0)->size(); i++) {
    ASSERT_EQ(right_answer.get()[i], layer_output.get()[i]);
  }
}