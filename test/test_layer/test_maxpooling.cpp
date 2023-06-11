#include <float.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/maxpooling.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"
using namespace std;
using namespace kuiper_infer;

std::shared_ptr<float> MaxPooling(std::shared_ptr<float> inputs, const int nums,
                                  const int channels, const int height,
                                  const int width, const int output_h,
                                  const int output_w, const int kernel_h,
                                  const int kernel_w, const int stride_h,
                                  const int stride_w, const int pad_h,
                                  const int pad_w) {
  float* output = new float[(nums * channels * output_h * output_w)];

  for (int n = 0; n < nums; n++) {
    for (int c = 0; c < channels; c++) {
      for (int ph = 0; ph < output_h; ph++) {
        for (int pw = 0; pw < output_w; pw++) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          const int hend = min(hstart + kernel_h, height);
          const int wend = min(wstart + kernel_w, width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          float maxval = -FLT_MAX;
          int maxidx = -1;

          const float* const input_slice =
              inputs.get() + (n * channels + c) * height * width;

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              if (input_slice[h * width + w] > maxval) {
                maxidx = h * width + w;
                maxval = input_slice[maxidx];
              }
            }
          }
          output[n * channels * output_h * output_w + c * output_h * output_w +
                 ph * output_w + pw] = maxval;
        }
      }
    }
  }
  std::shared_ptr<float> outputs(output, [](float* ptr) {
    delete[] ptr;  // 使用delete[]释放数组内存
  });
  return outputs;
}

TEST(test_layer, forward_max_pooling) {
  using namespace kuiper_infer;

  uint32_t nums = 3;
  uint32_t channels = 3;
  uint32_t row = 244;
  uint32_t col = 244;
  uint32_t padding_h = 0;
  uint32_t padding_w = 0;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(nums, channels, row, col);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  input->Rand();

  inputs.push_back(input);
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;

  const uint32_t output_h = uint32_t(
      std::floor((int(row) - int(kernel_h) + 2 * padding_h) / stride_h + 1));
  const uint32_t output_w = uint32_t(
      std::floor((int(col) - int(kernel_w) + 2 * padding_w) / stride_w + 1));

  MaxPoolingLayer max_layer(padding_h, padding_w, kernel_h, kernel_w, stride_h,
                            stride_w);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  max_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 1);
  auto right_answer =
      MaxPooling(input->to_cpu_data(), nums, channels, row, col, output_h,
                 output_w, kernel_h, kernel_w, stride_h, stride_w, 0, 0);

  auto layer_output = outputs.at(0)->to_cpu_data();
  for (int i = 0; i < outputs.at(0)->size(); i++) {
    ASSERT_EQ(right_answer.get()[i], layer_output.get()[i]);
  }
}