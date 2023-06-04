#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/cat.hpp"
#include "../../source/layer/details/linear.hpp"
#include "../../source/layer/details/relu.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"
using namespace std;

TEST(test_layer, forward_relu) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(32, 224, 512);
  input->Rand();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  ReluLayer relu_layer;
  const auto status = relu_layer.Forward(inputs, outputs);
  //   ASSERT_EQ(status, InferStatus::kInferSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    uint32_t size = input_->size();

    auto input_data = input_->to_cpu_data();
    auto output_data = output_->to_cpu_data();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_data.get()[j], input_data.get()[j]);
    }
  }
}

TEST(test_layer, forward_linear_simple) {
  using namespace kuiper_infer;
  const uint32_t in_features = 30000;
  const uint32_t out_features = 1033;
  const uint32_t in_dims = 10;

  LinearLayer linear_layer(in_features, out_features, false);

  std::vector<float> weights(in_features * out_features, 1.f);

  linear_layer.set_weights(weights);

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, in_dims, in_features);

  input->Fill(1.f);

  std::shared_ptr<Tensor<float>> output =
      std::make_shared<Tensor<float>>(1, in_dims, out_features);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::cout << outputs.at(0)->size() << std::endl;
  for (int i = 0; i < outputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    auto output_data = output_->to_cpu_data();

    for (int j = 0; j < output_->size(); ++j) {
      ASSERT_EQ(output_data.get()[j], in_features);
    }
  }
}



TEST(test_layer, forward_concat) {
  using namespace kuiper_infer;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  auto tensor1 = std::make_shared<Tensor<float>>(1, 3, 4, 4);
  tensor1.get()->Fill(1.f);
  auto tensor2 = std::make_shared<Tensor<float>>(1, 4, 4, 4);
  tensor2.get()->Fill(2.f);

  auto tensor3 = std::make_shared<Tensor<float>>(1, 5, 4, 4);
  tensor3.get()->Fill(3.f);

  inputs.push_back(tensor1);
  inputs.push_back(tensor2);
  inputs.push_back(tensor3);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  CatLayer cat_layer(1);
  const auto status = cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(outputs.at(0)->size(), 1 * (3 + 4 + 5) * 16);
  auto rightshape = vector<uint32_t>{1, 12, 4, 4};
  for (int i = 0; i < rightshape.size(); i++) {
    ASSERT_EQ(outputs.at(0)->shapes().at(i), rightshape.at(i));
  }
  auto outputs_data = outputs.at(0)->to_cpu_data();

  for (int i = 0; i < 3 * 4 * 4; i++) {
    ASSERT_EQ(outputs_data.get()[i], 1.f);
  }
  for (int i = 3 * 4 * 4; i < 4 * 4 * 4; i++) {
    ASSERT_EQ(outputs_data.get()[i], 2.f);
  }
}
// TEST(test_layer, forward_linear_graph) {
//   using namespace kuiper_infer;

//     // 设置日志输出目录（可选）
//     // FLAGS_log_dir = "/path/to/log/directory";

//     // 设置日志级别（可选）

//   RuntimeGraph
//   graph("/home/supercb/mycode/mlsys/KuiperInfer/tmp/linear/linear_1305_2407.pnnx.param",
//                      "/home/supercb/mycode/mlsys/KuiperInfer/tmp/linear/linear_1305_2407.pnnx.bin");

//   graph.Build("pnnx_input_0", "pnnx_output_0");
//   const uint32_t batch_size = 4;
//   std::vector<sftensor> inputs;

//   // for (uint32_t i = 0; i < batch_size; ++i) {
//     sftensor input = std::make_shared<ftensor>(4,1, 31, 1305);
//     input->Fill(1.f);
//     inputs.push_back(input);
//   // }
//   std::vector<sftensor> outputs;
//   // outputs = graph.Forward(inputs);
//   // arma::fmat real_data =
//   //     CSVDataLoader::LoadData("tmp/linear/linear_1305x2047.csv");
//   // for (const auto& output : outputs) {
//   //   for (uint32_t c = 0; c < output->channels(); ++c) {
//   //     bool is_same =
//   //         arma::approx_equal(real_data, output->slice(c), "absdiff",
//   1e-4f);
//   //     ASSERT_EQ(is_same, true);
//   //   }
//   // }
// }
