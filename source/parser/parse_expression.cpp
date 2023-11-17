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
#include "parser/parse_expression.hpp"
#include <glog/logging.h>
#include <algorithm>
#include <cctype>
#include <stack>
#include <utility>

namespace kuiper_infer {

void ReversePolish(const std::shared_ptr<TokenNode>& root_node,
                   std::vector<std::shared_ptr<TokenNode>>& reverse_polish) {
  if (root_node != nullptr) {
    ReversePolish(root_node->left, reverse_polish);
    ReversePolish(root_node->right, reverse_polish);
    reverse_polish.push_back(root_node);
  }
}

void ExpressionParser::Tokenizer(bool retokenize) {
  if (!retokenize && !this->tokens_.empty()) {
    return;
  }

  CHECK(!statement_.empty()) << "The input statement is empty!";
  statement_.erase(
      std::remove_if(statement_.begin(), statement_.end(), [](char c) { return std::isspace(c); }),
      statement_.end());
  CHECK(!statement_.empty()) << "The input statement is empty!";

  for (int32_t i = 0; i < statement_.size();) {
    char c = statement_.at(i);
    if (c == 'a') {
      CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'd')
          << "Parse add token failed, illegal character: " << statement_.at(i + 1);
      CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'd')
          << "Parse add token failed, illegal character: " << statement_.at(i + 2);
      Token token(TokenType::TokenAdd, i, i + 3);
      tokens_.push_back(token);
      std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
      token_strs_.push_back(token_operation);
      i = i + 3;
    } else if (c == 'm') {
      CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'u')
          << "Parse multiply token failed, illegal character: " << statement_.at(i + 1);
      CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'l')
          << "Parse multiply token failed, illegal character: " << statement_.at(i + 2);
      Token token(TokenType::TokenMul, i, i + 3);
      tokens_.push_back(token);
      std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
      token_strs_.push_back(token_operation);
      i = i + 3;
    } else if (c == '@') {
      CHECK(i + 1 < statement_.size() && std::isdigit(statement_.at(i + 1)))
          << "Parse number token failed, illegal character: " << statement_.at(i + 1);
      int32_t j = i + 1;
      for (; j < statement_.size(); ++j) {
        if (!std::isdigit(statement_.at(j))) {
          break;
        }
      }
      Token token(TokenType::TokenInputNumber, i, j);
      CHECK(token.start_pos < token.end_pos);
      tokens_.push_back(token);
      std::string token_input_number = std::string(statement_.begin() + i, statement_.begin() + j);
      token_strs_.push_back(token_input_number);
      i = j;
    } else if (c == ',') {
      Token token(TokenType::TokenComma, i, i + 1);
      tokens_.push_back(token);
      std::string token_comma = std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_comma);
      i += 1;
    } else if (c == '(') {
      Token token(TokenType::TokenLeftBracket, i, i + 1);
      tokens_.push_back(token);
      std::string token_left_bracket =
          std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_left_bracket);
      i += 1;
    } else if (c == ')') {
      Token token(TokenType::TokenRightBracket, i, i + 1);
      tokens_.push_back(token);
      std::string token_right_bracket =
          std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_right_bracket);
      i += 1;
    } else {
      LOG(FATAL) << "Unknown  illegal character: " << c;
    }
  }
}

const std::vector<Token>& ExpressionParser::tokens() const { return this->tokens_; }

const std::vector<std::string>& ExpressionParser::token_str_array() const { return this->token_strs_; }

std::shared_ptr<TokenNode> ExpressionParser::Generate_(int32_t& index) {
  CHECK(index < this->tokens_.size());
  const auto current_token = this->tokens_.at(index);
  CHECK(current_token.token_type == TokenType::TokenInputNumber ||
        current_token.token_type == TokenType::TokenAdd ||
        current_token.token_type == TokenType::TokenMul);
  if (current_token.token_type == TokenType::TokenInputNumber) {
    uint32_t start_pos = current_token.start_pos + 1;
    uint32_t end_pos = current_token.end_pos;
    CHECK(end_pos > start_pos || end_pos <= this->statement_.length())
        << "Current token has a wrong length";
    const std::string& str_number =
        std::string(this->statement_.begin() + start_pos, this->statement_.begin() + end_pos);
    return std::make_shared<TokenNode>(std::stoi(str_number), nullptr, nullptr);

  } else if (current_token.token_type == TokenType::TokenMul ||
             current_token.token_type == TokenType::TokenAdd) {
    std::shared_ptr<TokenNode> current_node = std::make_shared<TokenNode>();
    current_node->num_index = int32_t(current_token.token_type);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing left bracket!";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenLeftBracket);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing correspond left token!";
    const auto left_token = this->tokens_.at(index);

    if (left_token.token_type == TokenType::TokenInputNumber ||
        left_token.token_type == TokenType::TokenAdd ||
        left_token.token_type == TokenType::TokenMul) {
      current_node->left = Generate_(index);
    } else {
      LOG(FATAL) << "Unknown token type: " << int32_t(left_token.token_type);
    }

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing comma!";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenComma);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing correspond right token!";
    const auto right_token = this->tokens_.at(index);
    if (right_token.token_type == TokenType::TokenInputNumber ||
        right_token.token_type == TokenType::TokenAdd ||
        right_token.token_type == TokenType::TokenMul) {
      current_node->right = Generate_(index);
    } else {
      LOG(FATAL) << "Unknown token type: " << int32_t(right_token.token_type);
    }

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing right bracket!";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenRightBracket);
    return current_node;
  } else {
    LOG(FATAL) << "Unknown token type: " << int32_t(current_token.token_type);
  }
}

std::vector<std::shared_ptr<TokenNode>> ExpressionParser::Generate() {
  if (this->tokens_.empty()) {
    this->Tokenizer(true);
  }
  int32_t index = 0;
  std::shared_ptr<TokenNode> root = Generate_(index);
  CHECK(root != nullptr);
  CHECK(index == tokens_.size() - 1);

  // 转逆波兰式,之后转移到expression中
  std::vector<std::shared_ptr<TokenNode>> reverse_polish;
  ReversePolish(root, reverse_polish);

  return reverse_polish;
}

TokenNode::TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left,
                     std::shared_ptr<TokenNode> right)
    : num_index(num_index), left(left), right(right) {}
}  // namespace kuiper_infer