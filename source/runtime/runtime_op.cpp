//
// Created by fss on 22-11-28.
//
#include "runtime/runtime_op.hpp"
namespace kuiper_infer {
RuntimeOperator::~RuntimeOperator() {
  for (const auto &param : this->params) {
      delete param.second;
  }
}
}
