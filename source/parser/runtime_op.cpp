//
// Created by fss on 22-11-28.
//
#include "parser/runtime_op.hpp"
namespace kuiper_infer {
RuntimeOperator::~RuntimeOperator() {
  for (const auto &param : this->params) {
    if (param.second != nullptr) {
      delete param.second;
    }
  }
}
}
