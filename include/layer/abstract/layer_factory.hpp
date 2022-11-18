//
// Created by fss on 22-11-17.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_LAYER_FACTORY_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_LAYER_FACTORY_HPP_
#include <map>
#include <string>
#include <memory>
#include "layer.hpp"

namespace kuiper_infer {
class LayerRegisterer {
 public:
  typedef ParseParameterAttrStatus (*Creator)(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &layer);

  typedef std::map<std::string, Creator> CreateRegistry;

  static void RegisterCreator(const std::string &layer_type, const Creator &creator);

  static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator> &op);

  static CreateRegistry &Registry();
};

class LayerRegistererWrapper {
 public:
  LayerRegistererWrapper(const std::string &layer_type, const LayerRegisterer::Creator &creator) {
    LayerRegisterer::RegisterCreator(layer_type, creator);
  }
};

}

#endif //KUIPER_COURSE_SOURCE_LAYER_LAYER_FACTORY_HPP_
