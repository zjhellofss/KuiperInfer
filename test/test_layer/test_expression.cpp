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

// Created by fss on 22-12-1.
#include <gtest/gtest.h>
#include "../../source/layer/details/expression.hpp"
#include "data/load_data.hpp"
#include "parser/parse_expression.hpp"
#include "runtime/runtime_ir.hpp"

TEST(test_layer, add1) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/add/resnet_add.pnnx.param", "tmp/add/resnet_add.pnnx.bin");

  graph.Build();

  const int batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 4, 4);
    input->Fill(1.);
    inputs.push_back(input);
  }
  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);
  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.get_outputs("pnnx_output_0");
  ASSERT_EQ(output_tensors.size(), 4);

  const auto& output1 = output_tensors.at(0)->slice(0);
  const auto& output2 = CSVDataLoader::LoadData<float>("tmp/add/1.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t j = 0; j < size; ++j) {
    ASSERT_LE(abs(output1.at(j) - output2.at(j)), 5e-6);
  }
}

TEST(test_layer, add2) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/add/resnet_add2.pnnx.param", "tmp/add/resnet_add2.pnnx.bin");

  graph.Build();

  const int batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 4, 4);
    input->Fill(1.);
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);
  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.get_outputs("pnnx_output_0");
  ASSERT_EQ(output_tensors.size(), 4);

  const auto& output1 = output_tensors.at(0)->slice(0);
  const auto& output2 = CSVDataLoader::LoadData<float>("tmp/add/3.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t j = 0; j < size; ++j) {
    ASSERT_LE(abs(output1.at(j) - output2.at(j)), 5e-6);
  }
}

TEST(test_layer, mul1) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/add/resnet_add3.pnnx.param", "tmp/add/resnet_add3.pnnx.bin");

  graph.Build();

  const int batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 4, 4);
    input->Fill(1.);
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);
  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.get_outputs("pnnx_output_0");
  ASSERT_EQ(output_tensors.size(), 4);

  const auto& output1 = output_tensors.at(0)->slice(0);
  const auto& output2 = CSVDataLoader::LoadData<float>("tmp/add/7.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t j = 0; j < size; ++j) {
    ASSERT_LE(abs(output1.at(j) - output2.at(j)), 5e-6);
  }
}

TEST(test_expression, add1) {
  using namespace kuiper_infer;
  const std::string& str = "add(@0,@1)";
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
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(2.f);

  std::shared_ptr<Tensor<float>> output1 = outputs.front();
  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_expression, add2) {
  using namespace kuiper_infer;
  const std::string& str = "add(@0,@1)";
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
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(5.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_expression, mul1) {
  using namespace kuiper_infer;
  const std::string& str = "mul(@0,@1)";
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
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(6.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_expression, complex1) {
  using namespace kuiper_infer;
  const std::string& str = "mul(@2,add(@0,@1))";
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
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(20.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_expression, complex2) {
  using namespace kuiper_infer;
  const std::string& str = "mul(add(@0,@1),add(@2,@3))";
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
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(40.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_expression, complex3) {
  using namespace kuiper_infer;
  const std::string& str = "mul(mul(@0,@1),add(@2,@3))";
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
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(48.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_expression, complex4) {
  using namespace kuiper_infer;
  const std::string& str = "mul(mul(@0,@1), mul(@2,@3))";
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
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(96.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_expression, complex5) {
  using namespace kuiper_infer;
  const std::string& str = "mul(mul(@0,@1), mul(@2,add(@3,@4)))";
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
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(54.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_expression, complex6) {
  using namespace kuiper_infer;
  const std::string& str = "mul(mul(@0,@1), mul(@2,mul(@3,@4)))";
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
  ASSERT_EQ(status, StatusCode::kSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 = std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(120.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_parser, tokenizer) {
  using namespace kuiper_infer;
  const std::string& str = "add(add(add(@0,@1),@1),add(@0,@2))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto& tokens = parser.tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto& token_strs = parser.token_str_array();
  ASSERT_EQ(token_strs.empty(), false);

  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(tokens.at(0).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(tokens.at(1).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(2), "add");
  ASSERT_EQ(tokens.at(2).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(3), "(");
  ASSERT_EQ(tokens.at(3).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(4), "add");
  ASSERT_EQ(tokens.at(4).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(5), "(");
  ASSERT_EQ(tokens.at(5).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(6), "@0");
  ASSERT_EQ(tokens.at(6).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(tokens.at(7).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(8), "@1");
  ASSERT_EQ(tokens.at(8).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(9), ")");
  ASSERT_EQ(tokens.at(9).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(10), ",");
  ASSERT_EQ(tokens.at(10).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(11), "@1");
  ASSERT_EQ(tokens.at(11).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(12), ")");
  ASSERT_EQ(tokens.at(12).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(13), ",");
  ASSERT_EQ(tokens.at(13).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(14), "add");
  ASSERT_EQ(tokens.at(14).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(15), "(");
  ASSERT_EQ(tokens.at(15).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(16), "@0");
  ASSERT_EQ(tokens.at(16).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(17), ",");
  ASSERT_EQ(tokens.at(17).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(18), "@2");
  ASSERT_EQ(tokens.at(18).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(19), ")");
  ASSERT_EQ(tokens.at(19).token_type, TokenType::TokenRightBracket);
  ASSERT_EQ(token_strs.at(20), ")");
  ASSERT_EQ(tokens.at(20).token_type, TokenType::TokenRightBracket);
}

TEST(test_parser, tokenizer2) {
  using namespace kuiper_infer;
  const std::string& str = "add(add(add(@0,@1),@1),add(@0,add(@1,@1)))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto& tokens = parser.tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto& token_strs = parser.token_str_array();
  ASSERT_EQ(token_strs.empty(), false);

  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(tokens.at(0).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(tokens.at(1).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(2), "add");
  ASSERT_EQ(tokens.at(2).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(3), "(");
  ASSERT_EQ(tokens.at(3).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(4), "add");
  ASSERT_EQ(tokens.at(4).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(5), "(");
  ASSERT_EQ(tokens.at(5).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(6), "@0");
  ASSERT_EQ(tokens.at(6).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(tokens.at(7).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(8), "@1");
  ASSERT_EQ(tokens.at(8).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(9), ")");
  ASSERT_EQ(tokens.at(9).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(10), ",");
  ASSERT_EQ(tokens.at(10).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(11), "@1");
  ASSERT_EQ(tokens.at(11).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(12), ")");
  ASSERT_EQ(tokens.at(12).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(13), ",");
  ASSERT_EQ(tokens.at(13).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(14), "add");
  ASSERT_EQ(tokens.at(14).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(15), "(");
  ASSERT_EQ(tokens.at(15).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(16), "@0");
  ASSERT_EQ(tokens.at(16).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(17), ",");
  ASSERT_EQ(tokens.at(17).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(18), "add");
  ASSERT_EQ(tokens.at(18).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(19), "(");
  ASSERT_EQ(tokens.at(19).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(20), "@1");
  ASSERT_EQ(tokens.at(20).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(21), ",");
  ASSERT_EQ(tokens.at(21).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(22), "@1");
  ASSERT_EQ(tokens.at(22).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(23), ")");
  ASSERT_EQ(tokens.at(23).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(24), ")");
  ASSERT_EQ(tokens.at(24).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(25), ")");
  ASSERT_EQ(tokens.at(25).token_type, TokenType::TokenRightBracket);
}

TEST(test_parser, tokenizer3) {
  using namespace kuiper_infer;
  const std::string& str = "add(add(add(@0,@1),@2),mul(@0,@2))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto& tokens = parser.tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto& token_strs = parser.token_str_array();
  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(tokens.at(0).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(tokens.at(1).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(2), "add");
  ASSERT_EQ(tokens.at(2).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(3), "(");
  ASSERT_EQ(tokens.at(3).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(4), "add");
  ASSERT_EQ(tokens.at(4).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(5), "(");
  ASSERT_EQ(tokens.at(5).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(6), "@0");
  ASSERT_EQ(tokens.at(6).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(tokens.at(7).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(8), "@1");
  ASSERT_EQ(tokens.at(8).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(9), ")");
  ASSERT_EQ(tokens.at(9).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(10), ",");
  ASSERT_EQ(tokens.at(10).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(11), "@2");
  ASSERT_EQ(tokens.at(11).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(12), ")");
  ASSERT_EQ(tokens.at(12).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(13), ",");
  ASSERT_EQ(tokens.at(13).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(14), "mul");
  ASSERT_EQ(tokens.at(14).token_type, TokenType::TokenMul);

  ASSERT_EQ(token_strs.at(15), "(");
  ASSERT_EQ(tokens.at(15).token_type, TokenType::TokenLeftBracket);
}

TEST(test_parser, generate1) {
  using namespace kuiper_infer;
  const std::string& str = "add(@0,@1)";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto& nodes = parser.Generate();
  ASSERT_EQ(nodes.size(), 3);
  ASSERT_EQ(nodes.at(0)->num_index, 0);
  ASSERT_EQ(nodes.at(1)->num_index, 1);
  ASSERT_EQ(nodes.at(2)->num_index, int(TokenType::TokenAdd));
}

TEST(test_parser, generate2) {
  using namespace kuiper_infer;
  const std::string& str = "add(@0,add(@1,@2))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto& nodes = parser.Generate();
  ASSERT_EQ(nodes.size(), 5);
  ASSERT_EQ(nodes.at(0)->num_index, 0);
  ASSERT_EQ(nodes.at(1)->num_index, 1);
  ASSERT_EQ(nodes.at(2)->num_index, 2);

  ASSERT_EQ(nodes.at(3)->num_index, int(TokenType::TokenAdd));
  ASSERT_EQ(nodes.at(4)->num_index, int(TokenType::TokenAdd));
}

TEST(test_parser, generate3) {
  using namespace kuiper_infer;
  const std::string& str = "add(@0,add(@1,add(@3,@4)))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto& nodes = parser.Generate();
  ASSERT_EQ(nodes.size(), 7);
  ASSERT_EQ(nodes.at(0)->num_index, 0);
  ASSERT_EQ(nodes.at(1)->num_index, 1);
  ASSERT_EQ(nodes.at(2)->num_index, 3);

  ASSERT_EQ(nodes.at(3)->num_index, 4);
  ASSERT_EQ(nodes.at(4)->num_index, int(TokenType::TokenAdd));
  ASSERT_EQ(nodes.at(5)->num_index, int(TokenType::TokenAdd));
  ASSERT_EQ(nodes.at(6)->num_index, int(TokenType::TokenAdd));
}

TEST(test_parser, generate4) {
  using namespace kuiper_infer;
  const std::string& str = "add(@0,add(@1,add(@3,mul(@4,@5))))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto& nodes = parser.Generate();
  ASSERT_EQ(nodes.size(), 9);
  ASSERT_EQ(nodes.at(0)->num_index, 0);
  ASSERT_EQ(nodes.at(1)->num_index, 1);
  ASSERT_EQ(nodes.at(2)->num_index, 3);
  ASSERT_EQ(nodes.at(3)->num_index, 4);
  ASSERT_EQ(nodes.at(4)->num_index, 5);
  ASSERT_EQ(nodes.at(5)->num_index, -5);
  ASSERT_EQ(nodes.at(6)->num_index, -6);
  ASSERT_EQ(nodes.at(7)->num_index, -6);
  ASSERT_EQ(nodes.at(8)->num_index, -6);
}