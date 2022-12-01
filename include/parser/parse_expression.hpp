//
// Created by fss on 22-12-1.
//

#ifndef KUIPER_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_
#include <string>
#include <utility>
#include <vector>

namespace kuiper_infer {

enum class TokenType {
  TokenUnknown = -1,
  TokenInputNumber = 0,
  TokenComma = 1,
  TokenOperation = 2,
  TokenLeftBracket = 3,
  TokenRightBracket = 4,
};

struct Token {
  TokenType token_type = TokenType::TokenUnknown;
  int32_t start_pos = 0; //词语开始的位置
  int32_t end_pos = 0; // 词语结束的位置
  Token(TokenType token_type, int32_t start_pos, int32_t end_pos)
      : token_type(token_type), start_pos(start_pos), end_pos(end_pos) {

  }
};

// add(add(add(@0,@1),@1),add(@0,@2))
class ExpressionParser {
 public:
  explicit ExpressionParser(std::string statement) : statement_(std::move(statement)) {

  }

  void Parse();

  void Tokenizer();

  const std::vector<Token> &tokens() const;

  const std::vector<std::string> &token_strs() const;

 private:
  std::vector<Token> tokens_;
  std::vector<std::string> token_strs_;
  std::string statement_;
};
}

#endif //KUIPER_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_
