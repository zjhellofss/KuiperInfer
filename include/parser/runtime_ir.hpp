#include <vector>
#include <string>
#include <glog/logging.h>
#include <memory>
#include <map>

#include "ir.h"
#include "layer.hpp"

namespace kuiper_infer {
enum class DataType {
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

enum class ParameterType {
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
  explicit RuntimeParameter(ParameterType type = ParameterType::kParameterNull) : type(type) {

  }
  ParameterType type = ParameterType::kParameterNull;
};

struct RuntimeOperand {
  DataType type;
  std::string name;
  std::vector<int32_t> shapes;
};

struct RuntimeAttribute {
  // ir中的层权重参数
  std::vector<char> weight_data;
  std::vector<int> shape;
  DataType type = DataType::kTypeNull;

  template<class T>
  std::vector<T> get();
};

template<class T>
std::vector<T> RuntimeAttribute::get() {
  CHECK(!weight_data.empty());
  CHECK(type != DataType::kTypeNull);
  std::vector<T> weights;
  switch (type) {
    case DataType::kTypeFloat32: {
      const uint32_t float_size = 4;
      CHECK_EQ(weight_data.size() % float_size, 0);
      weights = std::vector<float>((float *) weight_data.data(),
                                   (float *) weight_data.data() + (weight_data.size() / float_size));
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
  std::string type;
  std::string name;
  std::vector<std::shared_ptr<RuntimeOperand> > inputs; // 操作数输入
  std::vector<std::shared_ptr<RuntimeOperand>> outputs; // 操作数输出

  // 该层算子中的配置和权重
  std::map<std::string, RuntimeParameter *> params;  ///算子的配置信息
  std::map<std::string, std::shared_ptr<RuntimeAttribute> > attribute; /// 算子的权重信息
};

struct RuntimeParameterInt : public RuntimeParameter {
  RuntimeParameterInt() : RuntimeParameter(ParameterType::kParameterInt) {

  }
  int value = 0;
};

struct RuntimeParameterFloat : public RuntimeParameter {
  RuntimeParameterFloat() : RuntimeParameter(ParameterType::kParameterFloat) {

  }
  float value = 0.f;
};

struct RuntimeParameterString : public RuntimeParameter {
  RuntimeParameterString() : RuntimeParameter(ParameterType::kParameterString) {

  }
  std::string value;
};

struct RuntimeParameterIntArray : public RuntimeParameter {
  RuntimeParameterIntArray() : RuntimeParameter(ParameterType::kParameterIntArray) {

  }
  std::vector<int> value;
};

struct RuntimeParameterFloatArray : public RuntimeParameter {
  RuntimeParameterFloatArray() : RuntimeParameter(ParameterType::kParameterFloatArray) {

  }
  std::vector<float> value;
};

struct RuntimeParameterStringArray : public RuntimeParameter {
  RuntimeParameterStringArray() : RuntimeParameter(ParameterType::kParameterStringArray) {

  }
  std::vector<std::string> value;
};

struct RuntimeParameterBool : public RuntimeParameter {
  RuntimeParameterBool() : RuntimeParameter(ParameterType::kParameterBool) {

  }
  bool value = false;
};

class RuntimeGraph {
 public:
  RuntimeGraph(const std::string &param_path, const std::string &bin_path);

  bool Init();

  void BuildLayers();

 private:
  uint32_t start_node_ = 0;
  std::string param_path_;
  std::string bin_path_;
  std::vector<std::shared_ptr<Layer>> layers_;

  std::vector<std::shared_ptr<RuntimeOperator>> input_operators_;
  std::vector<std::shared_ptr<RuntimeOperator>> operators_;
  std::unique_ptr<pnnx::Graph> graph_;
};
}