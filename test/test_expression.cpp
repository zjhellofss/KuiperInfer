//
// Created by fss on 22-12-1.
//
#include <gtest/gtest.h>
#include "parser/parse_expression.hpp"
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"
#include "../source/layer/details/expression.hpp"

TEST(test_expression, add1) {
  using namespace kuiper_infer;
  RuntimeGraph graph
      ("tmp/add/resnet_add.pnnx.param",
       "tmp/add/resnet_add.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  const int batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 4, 4);
    input->Fill(1.);
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  ASSERT_EQ(output_tensors.size(), 4);

  const auto &output1 = output_tensors.at(0)->at(0);
  const auto &output2 = CSVDataLoader::LoadData("tmp/add/1.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t j = 0; j < size; ++j) {
    ASSERT_LE(abs(output1.at(j) - output2.at(j)), 5e-6);
  }
}

TEST(test_expression, add2) {
  using namespace kuiper_infer;
  RuntimeGraph graph
      ("tmp/add/resnet_add2.pnnx.param",
       "tmp/add/resnet_add2.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  const int batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 4, 4);
    input->Fill(1.);
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  ASSERT_EQ(output_tensors.size(), 4);

  const auto &output1 = output_tensors.at(0)->at(0);
  const auto &output2 = CSVDataLoader::LoadData("tmp/add/3.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t j = 0; j < size; ++j) {
    ASSERT_LE(abs(output1.at(j) - output2.at(j)), 5e-6);
  }
}

TEST(test_expression, mul1) {
  using namespace kuiper_infer;
  RuntimeGraph graph
      ("tmp/add/resnet_add3.pnnx.param",
       "tmp/add/resnet_add3.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  const int batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 4, 4);
    input->Fill(1.);
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  ASSERT_EQ(output_tensors.size(), 4);

  const auto &output1 = output_tensors.at(0)->at(0);
  const auto &output2 = CSVDataLoader::LoadData("tmp/add/7.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t j = 0; j < size; ++j) {
    ASSERT_LE(abs(output1.at(j) - output2.at(j)), 5e-6);
  }
}

TEST(test_layer, add1) {
  using namespace kuiper_infer;
  const std::string &str = "add(@0,@1)";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(1.f);
  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(1.f);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(2.f);

  std::shared_ptr<Tensor<float>> output1 = outputs.front();
  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, add2) {
  using namespace kuiper_infer;
  const std::string &str = "add(@0,@1)";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(2.f);
  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(3.f);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(5.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, mul1) {
  using namespace kuiper_infer;
  const std::string &str = "mul(@0,@1)";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(2.f);
  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(3.f);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(6.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex1) {
  using namespace kuiper_infer;
  const std::string &str = "mul(@2,add(@0,@1))";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(2.f);
  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(3.f);

  std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(3, 224, 224);
  input3->Fill(4.f);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(20.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex2) {
  using namespace kuiper_infer;
  const std::string &str = "mul(add(@0,@1),add(@2,@3))";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(2.f);
  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(3.f);

  std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(3, 224, 224);
  input3->Fill(4.f);

  std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(3, 224, 224);
  input4->Fill(4.f);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(40.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex3) {
  using namespace kuiper_infer;
  const std::string &str = "mul(mul(@0,@1),add(@2,@3))";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(2.f);
  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(3.f);

  std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(3, 224, 224);
  input3->Fill(4.f);

  std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(3, 224, 224);
  input4->Fill(4.f);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(48.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex4) {
  using namespace kuiper_infer;
  const std::string &str = "mul(mul(@0,@1), mul(@2,@3))";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(2.f);
  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(3.f);

  std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(3, 224, 224);
  input3->Fill(4.f);

  std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(3, 224, 224);
  input4->Fill(4.f);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(96.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex5) {
  using namespace kuiper_infer;
  const std::string &str = "mul(mul(@0,@1), mul(@2,add(@3,@4)))";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(1.f);
  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(2.f);

  std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(3, 224, 224);
  input3->Fill(3.f);

  std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(3, 224, 224);
  input4->Fill(4.f);

  std::shared_ptr<Tensor<float>> input5 = std::make_shared<Tensor<float>>(3, 224, 224);
  input5->Fill(5.f);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);
  inputs.push_back(input5);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(54.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex6) {
  using namespace kuiper_infer;
  const std::string &str = "mul(mul(@0,@1), mul(@2,mul(@3,@4)))";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(1.f);
  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(2.f);

  std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(3, 224, 224);
  input3->Fill(3.f);

  std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(3, 224, 224);
  input4->Fill(4.f);

  std::shared_ptr<Tensor<float>> input5 = std::make_shared<Tensor<float>>(3, 224, 224);
  input5->Fill(5.f);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);
  inputs.push_back(input5);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(120.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_parser, tokenizer) {
  using namespace kuiper_infer;
  const std::string &str = "add(add(add(@0,@1),@1),add(@0,@2))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto &tokens = parser.tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto &token_strs = parser.token_strs();
  ASSERT_EQ(token_strs.empty(), false);

  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(token_strs.at(2), "add");
  ASSERT_EQ(token_strs.at(3), "(");
  ASSERT_EQ(token_strs.at(4), "add");

  ASSERT_EQ(token_strs.at(5), "(");
  ASSERT_EQ(token_strs.at(6), "@0");
  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(token_strs.at(8), "@1");
  ASSERT_EQ(token_strs.at(9), ")");

  ASSERT_EQ(token_strs.at(10), ",");
  ASSERT_EQ(token_strs.at(11), "@1");

  ASSERT_EQ(token_strs.at(12), ")");
  ASSERT_EQ(token_strs.at(13), ",");

  ASSERT_EQ(token_strs.at(14), "add");
  ASSERT_EQ(token_strs.at(15), "(");
  ASSERT_EQ(token_strs.at(16), "@0");
  ASSERT_EQ(token_strs.at(17), ",");
  ASSERT_EQ(token_strs.at(18), "@2");

  ASSERT_EQ(token_strs.at(19), ")");
  ASSERT_EQ(token_strs.at(20), ")");
}

TEST(test_parser, tokenizer2) {
  using namespace kuiper_infer;
  const std::string &str = "add(add(add(@0,@1),@1),add(@0,add(@1,@1)))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto &tokens = parser.tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto &token_strs = parser.token_strs();
  ASSERT_EQ(token_strs.empty(), false);

  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(token_strs.at(2), "add");
  ASSERT_EQ(token_strs.at(3), "(");
  ASSERT_EQ(token_strs.at(4), "add");

  ASSERT_EQ(token_strs.at(5), "(");
  ASSERT_EQ(token_strs.at(6), "@0");
  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(token_strs.at(8), "@1");
  ASSERT_EQ(token_strs.at(9), ")");

  ASSERT_EQ(token_strs.at(10), ",");
  ASSERT_EQ(token_strs.at(11), "@1");
  ASSERT_EQ(token_strs.at(12), ")");
  ASSERT_EQ(token_strs.at(13), ",");
  ASSERT_EQ(token_strs.at(14), "add");

  ASSERT_EQ(token_strs.at(15), "(");
  ASSERT_EQ(token_strs.at(16), "@0");
  ASSERT_EQ(token_strs.at(17), ",");
  ASSERT_EQ(token_strs.at(18), "add");
  ASSERT_EQ(token_strs.at(19), "(");

  ASSERT_EQ(token_strs.at(20), "@1");
  ASSERT_EQ(token_strs.at(21), ",");
  ASSERT_EQ(token_strs.at(22), "@1");
  ASSERT_EQ(token_strs.at(23), ")");
  ASSERT_EQ(token_strs.at(24), ")");
  ASSERT_EQ(token_strs.at(25), ")");
}

TEST(test_parser, tokenizer3) {
  using namespace kuiper_infer;
  const std::string &str = "add(add(add(@0,@1),@2),mul(@0,@2))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto &tokens = parser.tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto &token_strs = parser.token_strs();
  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(token_strs.at(2), "add");
  ASSERT_EQ(token_strs.at(3), "(");
  ASSERT_EQ(token_strs.at(4), "add");
  ASSERT_EQ(token_strs.at(5), "(");

  ASSERT_EQ(token_strs.at(6), "@0");
  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(token_strs.at(8), "@1");
  ASSERT_EQ(token_strs.at(9), ")");
  ASSERT_EQ(token_strs.at(10), ",");

  ASSERT_EQ(token_strs.at(11), "@2");
  ASSERT_EQ(token_strs.at(12), ")");
  ASSERT_EQ(token_strs.at(13), ",");
  ASSERT_EQ(token_strs.at(14), "mul");
  ASSERT_EQ(token_strs.at(15), "(");
}