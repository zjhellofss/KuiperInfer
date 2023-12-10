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

// Created by fss on 23-2-2.
#include <algorithm>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "../source/layer/details/softmax.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"
#include "tick.hpp"

// python ref https://pytorch.org/hub/pytorch_vision_resnet/
kuiper_infer::sftensor PreProcessImage(const cv::Mat& image) {
  using namespace kuiper_infer;
  assert(!image.empty());
  // 调整输入大小
  cv::Mat resize_image;
  cv::resize(image, resize_image, cv::Size(224, 224));

  cv::Mat rgb_image;
  cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

  rgb_image.convertTo(rgb_image, CV_32FC3);
  std::vector<cv::Mat> split_images;
  cv::split(rgb_image, split_images);
  uint32_t input_w = 224;
  uint32_t input_h = 224;
  uint32_t input_c = 3;
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

  uint32_t index = 0;
  for (const auto& split_image : split_images) {
    assert(split_image.total() == input_w * input_h);
    const cv::Mat& split_image_t = split_image.t();
    memcpy(input->slice(index).memptr(), split_image_t.data, sizeof(float) * split_image.total());
    index += 1;
  }

  float mean_r = 0.485f;
  float mean_g = 0.456f;
  float mean_b = 0.406f;

  float var_r = 0.229f;
  float var_g = 0.224f;
  float var_b = 0.225f;
  assert(input->channels() == 3);
  input->data() = input->data() / 255.f;
  input->slice(0) = (input->slice(0) - mean_r) / var_r;
  input->slice(1) = (input->slice(1) - mean_g) / var_g;
  input->slice(2) = (input->slice(2) - mean_b) / var_b;
  return input;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("usage: ./resnet_test [image path]\n");
    exit(-1);
  }
  using namespace kuiper_infer;

  const std::string& path = argv[1];

  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs;
  for (uint32_t i = 0; i < batch_size; ++i) {
    cv::Mat image = cv::imread(path);
    // 图像预处理
    sftensor input = PreProcessImage(image);
    inputs.push_back(input);
  }
  /**
  https://cowtransfer.com/s/a10a87ffd40640 点击链接查看 [ mobilenet_v2.pnnx.param ] ，
  或访问奶牛快传 cowtransfer.com 输入传输口令 951tw8 查看；
  */
  const std::string& param_path = "tmp/mobilenet/mobilenet_v2.pnnx.param";
  const std::string& weight_path = "tmp/mobilenet/mobilenet_v2.pnnx.bin";
  RuntimeGraph graph(param_path, weight_path);
  graph.Build();
  graph.set_inputs("pnnx_input_0", inputs);

  // 推理
  TICK(forward)
  graph.Forward(false);
  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");
  TOCK(forward)

  assert(outputs.size() == batch_size);
  // softmax
  std::vector<sftensor> outputs_softmax(batch_size);
  SoftmaxLayer softmax_layer(0);
  softmax_layer.Forward(outputs, outputs_softmax);
  assert(outputs_softmax.size() == batch_size);

  for (int i = 0; i < outputs_softmax.size(); ++i) {
    const sftensor& output_tensor = outputs_softmax.at(i);
    assert(output_tensor->size() == 1 * 1000);
    // 找到类别概率最大的种类
    float max_prob = -1;
    int max_index = -1;
    for (int j = 0; j < output_tensor->size(); ++j) {
      float prob = output_tensor->index(j);
      if (max_prob <= prob) {
        max_prob = prob;
        max_index = j;
      }
    }
    printf("class with max prob by mobilenet is %f index %d\n", max_prob, max_index);
  }
  return 0;
}