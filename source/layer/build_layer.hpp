//
// Created by fss on 22-11-15.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_BUILD_LAYER_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_BUILD_LAYER_HPP_
#include <memory>

class Layer;
class RuntimeOperator;
namespace kuiper_infer {

std::shared_ptr<Layer> BuildLayer(const std::string &layer_type, const std::shared_ptr<RuntimeOperator> &op);

}
#endif //KUIPER_COURSE_SOURCE_LAYER_BUILD_LAYER_HPP_
