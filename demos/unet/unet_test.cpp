#include <algorithm>
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../source/layer/details/softmax.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"
#include "tick.hpp"

kuiper_infer::sftensor PreProcessImage(const cv::Mat& image) {
  using namespace kuiper_infer;
  assert(!image.empty());
  cv::Mat resize_image;
  cv::resize(image, resize_image, cv::Size(512, 512));

  cv::Mat rgb_image;
  cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

  rgb_image.convertTo(rgb_image, CV_32FC3);
  std::vector<cv::Mat> split_images;
  cv::split(rgb_image, split_images);
  uint32_t input_w = 512;
  uint32_t input_h = 512;
  uint32_t input_c = 3;
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

  uint32_t index = 0;
  for (const auto& split_image : split_images) {
    assert(split_image.total() == input_w * input_h);
    const cv::Mat& split_image_t = split_image.t();
    memcpy(input->slice(index).memptr(), split_image_t.data, sizeof(float) * split_image.total());
    index += 1;
  }

  assert(input->channels() == 3);
  input->data() = input->data() / 255.f;
  return input;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("usage: ./unet_test [image path] [pnnx_param path] [pnnx_bin path]\n");
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

  const std::string& param_path = argv[2];
  const std::string& weight_path = argv[3];
  RuntimeGraph graph(param_path, weight_path);
  graph.Build();
  graph.set_inputs("pnnx_input_0", inputs);
  std::cout << "start inference!" << std::endl;
  TICK(forward)
  graph.Forward(true);
  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");
  TOCK(forward)
  assert(outputs.size() == batch_size);

  uint32_t input_h = 512;
  uint32_t input_w = 512;
  for (int i = 0; i < outputs.size(); ++i) {
    const sftensor& output_tensor = outputs.at(i);
    arma::fmat& out_channel_0 = output_tensor->slice(0);
    arma::fmat& out_channel_1 = output_tensor->slice(1);
    arma::fmat out_channel(input_h, input_w);
    assert(out_channel_0.size() == out_channel_1.size());
    assert(out_channel_0.size() == out_channel.size());

    for (int j = 0; j < out_channel_0.size(); j++) {
      if (out_channel_0.at(j) < out_channel_1.at(j)) {
        out_channel.at(j) = 255;
      } else {
        out_channel.at(j) = 0;
      }
    }

    arma::fmat out_channel_t = out_channel.t();
    auto output_array_ptr = out_channel_t.memptr();
    assert(output_array_ptr != nullptr);

    int data_type = CV_32F;
    cv::Mat output(input_h, input_w, data_type, output_array_ptr);
    cv::imwrite(cv::String("unet_output.jpg"), output);
  }

  return 0;
}