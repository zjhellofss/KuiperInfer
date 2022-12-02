//
// Created by fss on 22-11-28.
//

#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <string>
#include "layer/abstract/layer.hpp"
#include "runtime_operand.hpp"
#include "runtime_attr.hpp"
#include "runtime_parameter.hpp"

namespace kuiper_infer {
class Layer;
struct RuntimeOperator {
  int32_t meet_num = 0;
  ~RuntimeOperator();
  std::string type;
  std::string name;
  std::shared_ptr<Layer> layer; //节点对应的层

  std::vector<std::string> output_names;
  std::shared_ptr<RuntimeOperand> output_operands;
  std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;
  std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;
  std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators; //输出节点的名字和节点对应

  // 该层算子中的配置和权重
  std::map<std::string, RuntimeParameter *> params;  ///算子的配置信息
  std::map<std::string, std::shared_ptr<RuntimeAttribute> > attribute; /// 算子的权重信息
};
}
#endif //KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
