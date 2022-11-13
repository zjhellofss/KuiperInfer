//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_COURSE_INCLUDE_PARSER_GRAPH_HPP_
#define KUIPER_COURSE_INCLUDE_PARSER_GRAPH_HPP_
#include "storezip.hpp"
namespace kuiper_infer {
class Parameter {
 public:
  explicit Parameter()
      : type(0) {
  }

  explicit Parameter(bool b)
      : type(1), b(b) {
  }

  explicit Parameter(int i)
      : type(2), i(i) {
  }

  explicit Parameter(long l)
      : type(2), i(l) {
  }

  explicit Parameter(long long l)
      : type(2), i(l) {
  }

  explicit Parameter(float f)
      : type(3), f(f) {
  }

  explicit Parameter(double d)
      : type(3), f(d) {
  }

  explicit Parameter(const char *s)
      : type(4), str(s) {
  }

  explicit Parameter(const std::string &str)
      : type(4), str(str) {
  }

  Parameter(const std::initializer_list<int> &int_list)
      : type(5), int_array(int_list) {
  }

  Parameter(const std::initializer_list<int64_t> &int_list)
      : type(5) {
    for (const auto &x : int_list)
      int_array.push_back((int) x);
  }

  explicit Parameter(const std::vector<int> &int_array)
      : type(5), int_array(int_array) {
  }

  Parameter(const std::initializer_list<float> &float_array)
      : type(6), float_array(float_array) {
  }

  Parameter(const std::initializer_list<double> &double_array)
      : type(6) {
    for (const auto &x : double_array)
      float_array.push_back((float) x);
  }

  explicit Parameter(const std::vector<float> &float_array)
      : type(6), float_array(float_array) {
  }

  Parameter(const std::initializer_list<const char *> &str_list)
      : type(7) {
    for (const auto &x : str_list) {
      this->str_array.emplace_back(x);
    }
  }

  Parameter(const std::initializer_list<std::string> &str_array)
      : type(7), str_array(str_array) {
  }

  explicit Parameter(const std::vector<std::string> &str_array)
      : type(7), str_array(str_array) {
  }

  static Parameter parse_from_string(const std::string &value);

  // 0=null 1=b 2=i 3=f 4=str 5=int_array 6=float_array 7=str_array 8=others
  int type = 0;

  // value
  bool b = false;
  int i = 0;
  float f = 0.f;
  std::vector<int> int_array;
  std::vector<float> float_array;

  // keep std::string typed member the last for cross cxxabi compatibility
  std::string str;
  std::vector<std::string> str_array;
};

bool operator==(const Parameter &lhs, const Parameter &rhs);

class Attribute {
 public:
  Attribute()
      : type(0) {
  }

  Attribute(const std::initializer_list<int> &shape, const std::vector<float> &t);

  // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool
  int type;
  std::vector<int> shape;

  std::vector<char> data;
};

bool operator==(const Attribute &lhs, const Attribute &rhs);

// concat two attributes along the first axis
Attribute operator+(const Attribute &a, const Attribute &b);

class Operator;
class Operand {
 public:

  Operator *producer = nullptr;
  std::vector<Operator *> consumers;

  // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool 10=cp64 11=cp128 12=cp32
  int type = -1;
  std::vector<int> shape;

  // keep std::string typed member the last for cross cxxabi compatibility
  std::string name;
  std::map<std::string, Parameter> params;

 private:
  friend class Graph;
  Operand() = default;
};

class Operator {
 public:
  std::vector<Operand *> inputs;
  std::vector<Operand *> outputs;

  // keep std::string typed member the last for cross cxxabi compatibility
  std::string type;
  std::string name;

  std::vector<std::string> input_names;
  std::map<std::string, Parameter> params;
  std::map<std::string, Attribute> attrs;

 private:
  friend class Graph;
  Operator() = default;
};

class Graph {
 public:
  Graph();
  ~Graph();

  int load(const std::string &param_path, const std::string &binpath);

  Operator *new_operator(const std::string &type, const std::string &name);

  Operand *new_operand(const std::string &name);

  Operand *get_operand(const std::string &name);

  const Operand *get_operand(const std::string &name) const;

  std::vector<Operator *> ops;
  std::vector<Operand *> operands;
 private:
  Graph(const Graph &rhs);
  Graph &operator=(const Graph &rhs);
};
}
#endif //KUIPER_COURSE_INCLUDE_PARSER_GRAPH_HPP_
