#include <vector>
#include <string>
#include <glog/logging.h>
#include <memory>
#include <map>
#include <queue>

#include "ir.h"
#include "layer/abstract/layer.hpp"

namespace kuiper_infer {
enum class RuntimeDataType {
  kTypeNull = 0,
  kTypeFloat32 = 1,
  kTypeFloat64 = 2,
  kTypeFloat16 = 3,
  kTypeInt32 = 4,
  kTypeInt64 = 5,
  kTypeInt16 = 6,
  kTypeInt8 = 7,
  kTypeUInt8 = 8,
};

enum class RuntimeParameterType {
  kParameterNull = 0,
  kParameterInt = 1,
  kParameterFloat = 2,
  kParameterString = 3,
  kParameterIntArray = 4,
  kParameterFloatArray = 5,
  kParameterStringArray = 6,
  kParameterBool = 7,
};

struct RuntimeParameter {
  virtual ~RuntimeParameter() = default;

  explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::kParameterNull) : type(type) {

  }
  RuntimeParameterType type = RuntimeParameterType::kParameterNull;
};

struct RuntimeOperand {
  RuntimeDataType type;
  std::string name;
  std::vector<int32_t> shapes;
  std::vector<std::shared_ptr<Tensor>> datas;
};

struct RuntimeAttribute {
  // ir中的层权重参数
  std::vector<char> weight_data;
  std::vector<int> shape;
  RuntimeDataType type = RuntimeDataType::kTypeNull;

  template<class T>
  std::vector<T> get();
};

template<class T>
std::vector<T> RuntimeAttribute::get() {
  CHECK(!weight_data.empty());
  CHECK(type != RuntimeDataType::kTypeNull);
  std::vector<T> weights;
  switch (type) {
    case RuntimeDataType::kTypeFloat32: {
      const bool is_double = std::is_same<T, double>::value;
      CHECK_EQ(is_double, true);
      const uint32_t float_size = 4;
      CHECK_EQ(weight_data.size() % float_size, 0);
      for (int i = 0; i < weight_data.size() / float_size; ++i) {
        double v = *((float *) weight_data.data() + i);
        weights.push_back(v);
      }
      break;
    }
    default: {
      LOG(FATAL) << "Unknown weight_data Type";
    }
  }
  return weights;
}

struct RuntimeOperator {
  ~RuntimeOperator();
  int has_transfer = 0;
  std::string type;
  std::string name;
  std::shared_ptr<Layer> layer; //节点对应的层

  std::vector<std::string> output_names;
  std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;
  std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators; //输出节点的名字和节点对应

  // 该层算子中的配置和权重
  std::map<std::string, RuntimeParameter *> params;  ///算子的配置信息
  std::map<std::string, std::shared_ptr<RuntimeAttribute> > attribute; /// 算子的权重信息
};

struct RuntimeParameterInt : public RuntimeParameter {
  RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::kParameterInt) {

  }
  int value = 0;
};

struct RuntimeParameterFloat : public RuntimeParameter {
  RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::kParameterFloat) {

  }
  float value = 0.f;
};

struct RuntimeParameterString : public RuntimeParameter {
  RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::kParameterString) {

  }
  std::string value;
};

struct RuntimeParameterIntArray : public RuntimeParameter {
  RuntimeParameterIntArray() : RuntimeParameter(RuntimeParameterType::kParameterIntArray) {

  }
  std::vector<int> value;
};

struct RuntimeParameterFloatArray : public RuntimeParameter {
  RuntimeParameterFloatArray() : RuntimeParameter(RuntimeParameterType::kParameterFloatArray) {

  }
  std::vector<float> value;
};

struct RuntimeParameterStringArray : public RuntimeParameter {
  RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::kParameterStringArray) {

  }
  std::vector<std::string> value;
};

struct RuntimeParameterBool : public RuntimeParameter {
  RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::kParameterBool) {

  }
  bool value = false;
};

class RuntimeGraph {
 public:
  RuntimeGraph(const std::string &param_path, const std::string &bin_path);

  void Build();

  void set_bin_path(const std::string &bin_path);

  void set_param_path(const std::string &param_path);

  const std::string &param_path() const;

  const std::string &bin_path() const;

  std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_data,
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

  bool CheckDAG();

  static std::vector<std::shared_ptr<Tensor>> CloneData(const std::vector<std::shared_ptr<Tensor>> &data);

  static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator> &op);

  static bool CheckOperatorReady(const std::shared_ptr<RuntimeOperator> &op);

 private:
  enum class GraphState {
    NeedInit = -2,
    NeedBuild = -1,
    Complete = 0,
  };
  GraphState graph_state_ = GraphState::NeedInit;
  std::string param_path_;
  std::string bin_path_;
  std::vector<std::shared_ptr<RuntimeOperator>> input_operators_;
  std::vector<std::shared_ptr<RuntimeOperator>> output_operators_;
  std::vector<std::shared_ptr<RuntimeOperator>> operators_;
  std::unique_ptr<pnnx::Graph> graph_;
};

}