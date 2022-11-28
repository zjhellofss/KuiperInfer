//
// Created by fss on 22-11-28.
//

#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#include <vector>
#include <string>
#include <memory>
#include "common.hpp"
#include "data/tensor.hpp"

namespace kuiper_infer {
struct RuntimeOperand {
  RuntimeDataType type;
  std::string name;
  std::vector<int32_t> shapes;
  std::vector<std::shared_ptr<Tensor<float>>> datas;
};
}
#endif //KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
