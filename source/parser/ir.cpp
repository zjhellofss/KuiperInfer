//
// Created by fss on 22-11-13.
//
#include "parser/ir.hpp"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <stack>

#include <glog/logging.h>

namespace kuiper_infer {

static size_t TypeToElementSize(int type) {
  if (type == 1) return 4;
  if (type == 2) return 8;
  if (type == 3) return 2;
  if (type == 4) return 4;
  if (type == 5) return 8;
  if (type == 6) return 2;
  if (type == 7) return 1;
  if (type == 8) return 1;
  if (type == 9) return 1;
  if (type == 10) return 8;
  if (type == 11) return 16;
  if (type == 12) return 4;
  return 0; // null
}

static int StringToType(const char *s) {
  if (strcmp(s, "f32") == 0) return 1;
  if (strcmp(s, "f64") == 0) return 2;
  if (strcmp(s, "f16") == 0) return 3;
  if (strcmp(s, "i32") == 0) return 4;
  if (strcmp(s, "i64") == 0) return 5;
  if (strcmp(s, "i16") == 0) return 6;
  if (strcmp(s, "i8") == 0) return 7;
  if (strcmp(s, "u8") == 0) return 8;
  if (strcmp(s, "bool") == 0) return 9;
  if (strcmp(s, "cp64") == 0) return 10;
  if (strcmp(s, "cp128") == 0) return 11;
  if (strcmp(s, "cp32") == 0) return 12;
  return 0; // null
}

bool operator==(const Parameter &lhs, const Parameter &rhs) {
  if (lhs.type != rhs.type)
    return false;

  if (lhs.type == 0)
    return true;

  if (lhs.type == 1 && lhs.b == rhs.b)
    return true;

  if (lhs.type == 2 && lhs.i == rhs.i)
    return true;

  if (lhs.type == 3 && lhs.f == rhs.f)
    return true;

  if (lhs.type == 4 && lhs.str == rhs.str)
    return true;

  if (lhs.type == 5 && lhs.int_array == rhs.int_array)
    return true;

  if (lhs.type == 6 && lhs.float_array == rhs.float_array)
    return true;

  if (lhs.type == 7 && lhs.str_array == rhs.str_array)
    return true;

  return false;
}

Attribute::Attribute(const std::initializer_list<int> &_shape, const std::vector<float> &t) {
  type = 1;
  shape = _shape;

  if (!shape.empty()) {
    int size = shape[0];
    for (size_t i = 1; i < shape.size(); i++) {
      size *= shape[i];
    }

    data.resize(size * TypeToElementSize(type));
    memcpy((void *) data.data(), (const void *) t.data(), data.size());
  }
}

bool operator==(const Attribute &lhs, const Attribute &rhs) {
  if (lhs.type != rhs.type)
    return false;

  if (lhs.type == 0)
    return true;

  if (lhs.shape != rhs.shape)
    return false;

  if (lhs.data != rhs.data)
    return false;

  return true;
}

Attribute operator+(const Attribute &a, const Attribute &b) {
  Attribute c;

  if (a.type != b.type) {
    fprintf(stderr, "concat attribute type mismatch\n");
    return c;
  }

  if (a.shape.size() != b.shape.size()) {
    fprintf(stderr, "concat attribute shape rank mismatch\n");
    return c;
  }

  for (int i = 1; i < (int) a.shape.size(); i++) {
    if (a.shape[i] != b.shape[i]) {
      fprintf(stderr, "concat attribute shape mismatch\n");
      return c;
    }
  }

  c.type = a.type;
  c.shape = a.shape;
  c.shape[0] += b.shape[0]; // concat the first dim

  c.data.resize(a.data.size() + b.data.size());
  memcpy(c.data.data(), a.data.data(), a.data.size());
  memcpy(c.data.data() + a.data.size(), b.data.data(), b.data.size());

  return c;
}

Parameter Parameter::parse_from_string(const std::string &value) {
  Parameter p;
  p.type = 0;

  if (value == "None" || value == "()" || value == "[]") {
    return p;
  }

  if (value == "True" || value == "False") {
    // bool
    p.type = 1;
    p.b = value == "True";
    return p;
  }

  if (value[0] == '(' || value[0] == '[') {
    // list
    std::string lc = value.substr(1, value.size() - 2);
    std::istringstream lcss(lc);

    while (!lcss.eof()) {
      std::string elem;
      std::getline(lcss, elem, ',');

      if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9'))
          || (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9'))) {
        // string
        p.type = 7;
        p.str_array.push_back(elem);
      } else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos) {
        // float
        p.type = 6;
        p.float_array.push_back(std::stof(elem));
      } else {
        // integer
        p.type = 5;
        p.int_array.push_back(std::stoi(elem));
      }
    }
    return p;
  }

  if ((value[0] != '-' && (value[0] < '0' || value[0] > '9'))
      || (value[0] == '-' && (value[1] < '0' || value[1] > '9'))) {
    // string
    p.type = 4;
    p.str = value;
    return p;
  }

  if (value.find('.') != std::string::npos || value.find('e') != std::string::npos) {
    // float
    p.type = 3;
    p.f = std::stof(value);
    return p;
  }

  // integer
  p.type = 2;
  p.i = std::stoi(value);
  return p;
}

Graph::Graph() {
}

Graph::~Graph() {
  for (auto x : ops)
    delete x;

  for (auto x : operands)
    delete x;

  ops.clear();
  operands.clear();
}

Graph::Graph(const Graph &) {
}

Graph &Graph::operator=(const Graph &) {
  return *this;
}

static void load_parameter(Operator *op, const std::string &key, const std::string &value) {
  op->params[key] = Parameter::parse_from_string(value);
}

static void load_input_key(Operator *op, const std::string &key, const std::string &value) {
  op->input_names.resize(op->inputs.size());

  for (size_t i = 0; i < op->inputs.size(); i++) {
    const Operand *oprand = op->inputs[i];
    if (oprand->name == value) {
      op->input_names[i] = key;
      break;
    }
  }
}

static void load_shape(Operator *op, const std::string &key, const std::string &value) {
  Operand *operand = 0;
  for (auto r : op->inputs) {
    if (r->name == key) {
      operand = r;
      break;
    }
  }

  if (!operand) {
    for (auto r : op->outputs) {
      if (r->name == key) {
        operand = r;
        break;
      }
    }
  }

  if (!operand) {
    fprintf(stderr, "no such operand %str for operator %str\n", key.c_str(), op->name.c_str());
    return;
  }

  // type
  std::string type_str = value.substr(value.find_last_of(')') + 1);
  operand->type = StringToType(type_str.c_str());

  // shape
  std::string lc = value.substr(1, value.find_last_of(')') - 1);
  std::istringstream lcss(lc);

  operand->shape.clear();
  while (!lcss.eof()) {
    std::string elem;
    std::getline(lcss, elem, ',');

    if (elem == "?") {
      operand->shape.push_back(-1);
    } else {
      int i = std::stoi(elem);
      operand->shape.push_back(i);
    }
  }
}

static void load_attribute(Operator *op, const std::string &key, const std::string &value, StoreZipReader &szr) {
  Attribute &a = op->attrs[key];

  // type
  std::string typestr = value.substr(value.find_last_of(')') + 1);
  a.type = StringToType(typestr.c_str());

  if (a.type == 0)
    return;

  // shape
  std::string lc = value.substr(1, value.find_last_of(')') - 1);
  std::istringstream lcss(lc);

  a.shape.clear();
  while (!lcss.eof()) {
    std::string elem;
    std::getline(lcss, elem, ',');

    int i = std::stoi(elem);
    a.shape.push_back(i);
  }

  if (a.shape.empty())
    return;

  // data
  size_t size = 1;
  for (int i : a.shape) {
    size *= i;
  }

  size_t bytesize = size * TypeToElementSize(a.type);

  std::string filename = op->name + "." + key;

  size_t filesize = szr.get_file_size(filename);

  if (filesize == 0) {
    // no such file
    return;
  }

  if (filesize != bytesize) {
    fprintf(stderr, "file size not match expect %lu but got %lu\n", bytesize, filesize);
  }

  a.data.resize(bytesize);
  szr.read_file(filename, (char *) a.data.data());
}

int Graph::load(const std::string &param_path, const std::string &binpath) {
  std::ifstream is(param_path, std::ios::in | std::ios::binary);
  if (!is.good()) {
    LOG(ERROR) << "Open failed!";
    return -1;
  }

  StoreZipReader szr;
  if (szr.open(binpath) != 0) {
    fprintf(stderr, "open failed\n");
    return -1;
  }

  int magic = 0;
  {
    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);

    iss >> magic;
  }

  int operator_count = 0;
  int operand_count = 0;
  {
    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);

    iss >> operator_count >> operand_count;
  }

  for (int i = 0; i < operator_count; i++) {
    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);

    std::string type;
    std::string name;
    int input_count = 0;
    int output_count = 0;

    iss >> type >> name >> input_count >> output_count;

    Operator *op = new_operator(type, name);

    for (int j = 0; j < input_count; j++) {
      std::string operand_name;
      iss >> operand_name;

      Operand *r = get_operand(operand_name);
      if (r != nullptr) {
        r->consumers.push_back(op);
        op->inputs.push_back(r);
      }
    }

    for (int j = 0; j < output_count; j++) {
      std::string operand_name;
      iss >> operand_name;

      Operand *r = new_operand(operand_name);
      r->producer = op;
      op->outputs.push_back(r);
    }

    // key=value
    while (!iss.eof()) {
      std::string param;
      iss >> param;

      std::string key;
      std::string value;
      std::istringstream pss(param);
      std::getline(pss, key, '=');
      std::getline(pss, value);

      if (key[0] == '@') {
        // attribute
        load_attribute(op, key.substr(1), value, szr);
      } else if (key[0] == '$') {
        // operand input key
        load_input_key(op, key.substr(1), value);
      } else if (key[0] == '#') {
        // operand shape
        load_shape(op, key.substr(1), value);
      } else {
        // parameter
        load_parameter(op, key, value);
      }
    }
  }

  return 0;
}

Operator *Graph::new_operator(const std::string &type, const std::string &name) {
  Operator *op = new Operator;
  op->type = type;
  op->name = name;
  ops.push_back(op);
  return op;
}

Operand *Graph::new_operand(const std::string &name) {
  Operand *r = new Operand;
  r->name = name;
  operands.push_back(r);
  return r;
}

Operand *Graph::get_operand(const std::string &name) {
  for (Operand *r : operands) {
    if (r->name == name)
      return r;
  }
  return nullptr;
}

const Operand *Graph::get_operand(const std::string &name) const {
  for (const Operand *r : operands) {
    if (r->name == name)
      return r;
  }
  return nullptr;
}
}