//
// Created by fss on 22-12-1.
//
#include "parser/parse_expression.hpp"
#include <algorithm>
#include <cctype>
#include <glog/logging.h>

namespace kuiper_infer {

void ExpressionParser::Tokenizer() {
  CHECK(!statement_.empty()) << "The input statement is empty!";
  statement_.erase(std::remove_if(statement_.begin(), statement_.end(), [](char c) {
    return std::isspace(c);
  }), statement_.end());
  CHECK(!statement_.empty()) << "The input statement is empty!";

  for (int32_t i = 0; i < statement_.size();) {
    char c = statement_.at(i);
    if (c == 'a') {
      CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'd')
              << "Parse add token failed, illegal character: " << c;
      CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'd')
              << "Parse add token failed, illegal character: " << c;
      Token token(TokenType::TokenOperation, i, i + 3);
      tokens_.push_back(token);
      std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
      token_strs_.push_back(token_operation);
      i = i + 3;
    } else if (c == '@') {
      CHECK(i + 1 < statement_.size() && std::isdigit(statement_.at(i + 1)))
              << "Parse number token failed, illegal character: " << c;
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
      std::string token_left_bracket = std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_left_bracket);
      i += 1;
    } else if (c == ')') {
      Token token(TokenType::TokenRightBracket, i, i + 1);
      tokens_.push_back(token);
      std::string token_right_bracket = std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_right_bracket);
      i += 1;
    } else {
      LOG(FATAL) << "Unknown  illegal character: " << c;
    }
  }
}

const std::vector<Token> &ExpressionParser::tokens() const {
  return this->tokens_;
}

const std::vector<std::string> &ExpressionParser::token_strs() const {
  return this->token_strs_;
}

}