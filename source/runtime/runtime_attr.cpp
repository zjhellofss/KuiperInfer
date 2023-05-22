//
// Created by fss on 23-3-4.
//

#include "runtime/runtime_attr.hpp"
namespace kuiper_infer {
void RuntimeAttribute::ClearWeight() {
  if (!this->weight_data.empty()) {
    std::vector<char> tmp = std::vector<char>();
    this->weight_data.swap(tmp);
  }
}
}  // namespace kuiper_infer