//
// Created by fss on 22-12-1.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "parser/parse_expression.hpp"
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"

TEST(test_expression, add1) {
  using namespace kuiper_infer;
  RuntimeGraph graph
      ("tmp/add/resnet_batchnorm_add.pnnx.param",
       "tmp/add/resnet_batchnorm_add.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(1, 4, 4);
  input1->Fill(1.);

  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(1, 4, 4);
  input2->Fill(1.);

  std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(1, 4, 4);
  input3->Fill(1.);

  std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(1, 4, 4);
  input4->Fill(1.);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  ASSERT_EQ(output_tensors.size(), 4);

  const auto &output1 = output_tensors.at(0)->at(0);
//  std::cout << output1 << std::endl;
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
      ("tmp/add/resnet_batchnorm_add2.pnnx.param",
       "tmp/add/resnet_batchnorm_add2.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(1, 4, 4);
  input1->Fill(1.);

  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(1, 4, 4);
  input2->Fill(1.);

  std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(1, 4, 4);
  input3->Fill(1.);

  std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(1, 4, 4);
  input4->Fill(1.);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

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
      ("tmp/add/resnet_batchnorm_add3.pnnx.param",
       "tmp/add/resnet_batchnorm_add3.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(1, 4, 4);
  input1->Fill(1.);

  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(1, 4, 4);
  input2->Fill(1.);

  std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(1, 4, 4);
  input3->Fill(1.);

  std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(1, 4, 4);
  input4->Fill(1.);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

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
//TEST(test_parser, tokenizer_gen) {
//  using namespace kuiper_infer;
//  const std::string &str = "mul(add(add(@0,@1),@2),add(@0,@2))";
//  ExpressionParser parser(str);
//  parser.Tokenizer();
//  const auto &nodes = parser.Generate();
//  ASSERT_EQ(nodes->num_index, -3);
//  ASSERT_EQ(nodes->left->num_index, -2);
//  ASSERT_EQ(nodes->left->left->left->num_index, 0);
//
//  ASSERT_EQ(nodes->left->left->num_index, -2);
//  ASSERT_EQ(nodes->left->right->num_index, 2);
//
//  ASSERT_EQ(nodes->right->left->num_index, 0);
//  ASSERT_EQ(nodes->right->right->num_index, 2);
//  ASSERT_EQ(nodes->right->num_index, -2);
//}

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