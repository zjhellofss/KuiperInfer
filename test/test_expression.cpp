//
// Created by fss on 22-12-1.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "parser/parse_expression.hpp"

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