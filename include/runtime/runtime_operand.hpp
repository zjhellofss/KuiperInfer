//
// Created by fss on 22-11-28.
//

#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#include <vector>
#include <string>
#include <memory>
#include "status_code.hpp"
#include "runtime_datatype.hpp"
#include "data/tensor.hpp"

namespace kuiper_infer {
/// 计算节点输入输出的操作数
struct RuntimeOperand {
  std::string name; /// 操作数的名称
  std::vector<int32_t> shapes; /// 操作数的形状
  std::vector<std::shared_ptr<Tensor<float>>> datas; /// 存储操作数
  RuntimeDataType type = RuntimeDataType::kTypeUnknown; /// 操作数的类型，一般是float
};
}
#endif //KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
