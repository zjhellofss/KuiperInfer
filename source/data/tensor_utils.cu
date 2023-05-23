

#include <glog/logging.h>
#include "data/tensor.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer {
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols) {
  return std::make_shared<Tensor<float>>(channels, rows, cols);
}

std::shared_ptr<Tensor<float>> TensorCreate(
     std::vector<int32_t>& shapes) {
  CHECK(shapes.size() == 3);
  return TensorCreate(shapes.at(0), shapes.at(1), shapes.at(2));
}
}  // namespace kuiper_infer