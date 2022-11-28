#include <vector>
#include <string>
#include <glog/logging.h>
#include <memory>
#include <map>
#include <queue>

#include "ir.h"
#include "layer/abstract/layer.hpp"
#include "parser/runtime_operand.hpp"
#include "runtime_op.hpp"

namespace kuiper_infer {
class RuntimeGraphShape {
 public:
  static void InitOperatorInputTensor(const std::vector<std::shared_ptr<RuntimeOperator>> &operators);

  static void InitOperatorOutputTensor(const std::vector<pnnx::Operator *>& pnnx_operators,
                                       const std::vector<std::shared_ptr<RuntimeOperator>> &operators);
};

class RuntimeGraph {
 public:
  RuntimeGraph(std::string param_path, std::string bin_path);

  void Build(const std::string &input_name, const std::string &output_name);

  void set_bin_path(const std::string &bin_path);

  void set_param_path(const std::string &param_path);

  const std::string &param_path() const;

  const std::string &bin_path() const;

  std::vector<std::shared_ptr<Tensor<float>>> Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                                      bool debug = false);

 private:
  bool Init();

  static void InitInputOperators(const std::vector<pnnx::Operand *> &inputs,
                                 const std::shared_ptr<RuntimeOperator> &runtime_operator);

  static void InitOutputOperators(const std::vector<pnnx::Operand *> &outputs,
                                  const std::shared_ptr<RuntimeOperator> &runtime_operator);

  static void InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                             const std::shared_ptr<RuntimeOperator> &runtime_operator);

  static void InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                              const std::shared_ptr<RuntimeOperator> &runtime_operator);

  static void CloneData(const std::vector<std::shared_ptr<Tensor<float>>> &src,
                        const std::vector<std::shared_ptr<Tensor<float>>> &dest);

  static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator> &op);

  static bool CheckOperatorReady(const std::shared_ptr<RuntimeOperator> &op);

  static void ProbeNextLayer(const std::shared_ptr<RuntimeOperator> &current_op,
                             std::deque<std::shared_ptr<RuntimeOperator>> &operator_queue,
                             std::vector<std::shared_ptr<Tensor<float>>> &layer_output_data);

 private:
  enum class GraphState {
    NeedInit = -2,
    NeedBuild = -1,
    Complete = 0,
  };
  GraphState graph_state_ = GraphState::NeedInit;
  std::string input_name_;
  std::string output_name_;
  std::string param_path_;
  std::string bin_path_;
  std::map<std::string, std::shared_ptr<RuntimeOperator>> input_operators_maps_;
  std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators_maps_;
  std::vector<std::shared_ptr<RuntimeOperator>> operators_;
  std::unique_ptr<pnnx::Graph> graph_;
};

}