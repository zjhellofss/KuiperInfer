//
// Created by fss on 22-12-26.
//
#include "yolo_detect.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {

YoloDetectLayer::YoloDetectLayer(int32_t stages,
                                 int32_t num_classes,
                                 const std::vector<float> &strides,
                                 const std::vector<arma::fmat> &anchor_grids,
                                 const std::vector<arma::fmat> &grids,
                                 const std::vector<std::shared_ptr<ConvolutionLayer>> &conv_layers)
    : Layer("yolo"), stages_(stages), num_classes_(num_classes), strides_(strides), anchor_grids_(anchor_grids),
      grids_(grids), conv_layers_(conv_layers) {

}

InferStatus YoloDetectLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of yolo detect layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t stages = stages_;
  const uint32_t classes_info = num_classes_ + 5;
  const uint32_t input_size = inputs.size();
  const uint32_t batch_size = outputs.size();

  if (input_size / batch_size != stages_ || input_size % batch_size != 0) {
    LOG(ERROR) << "The input and output number of yolo detect layer is wrong";
    return InferStatus::kInferFailedYoloStageNumberError;
  }

  CHECK(!this->conv_layers_.empty() && this->conv_layers_.size() == stages);
  std::vector<std::vector<std::shared_ptr<Tensor<float>>>> batches(stages);
  for (uint32_t i = 0; i < input_size; ++i) {
    const uint32_t index = i / batch_size;
    batches.at(index).push_back(inputs.at(i));
  }

  uint32_t concat_rows = 0;
  std::vector<std::shared_ptr<Tensor<float>>> zs(stages);

#pragma omp parallel for num_threads(stages) reduction(+:concat_rows)
  for (uint32_t stage = 0; stage < stages; ++stage) {
    const std::vector<std::shared_ptr<Tensor<float>>> &stage_input = batches.at(stage);
    CHECK(stage_input.size() == batch_size);

    std::vector<std::shared_ptr<Tensor<float>>> stage_output(batch_size);
    const auto status = this->conv_layers_.at(stage)->Forward(stage_input, stage_output);
    CHECK(status == InferStatus::kInferSuccess);

    CHECK(stage_output.size() == batch_size);
    std::shared_ptr<Tensor<float>> x_stages_tensor;
    for (uint32_t b = 0; b < batch_size; ++b) {
      std::shared_ptr<Tensor<float>> input = stage_output.at(b);
      const uint32_t nx = input->rows();
      const uint32_t ny = input->cols();
      if (x_stages_tensor == nullptr || x_stages_tensor->empty()) {
        x_stages_tensor = std::make_shared<Tensor<float>>(batch_size, stages * nx * ny, uint32_t(classes_info));
      }

      input->ReRawView({stages, uint32_t(classes_info), ny * nx});
      input->Transform([](const float f) {
        return 1 / (1 + std::exp(-f));
      });

      arma::fmat &x_stages = x_stages_tensor->at(b);
      for (uint32_t s = 0; s < stages; ++s) {
        x_stages.submat(ny * nx * s, 0, ny * nx * (s + 1) - 1, classes_info - 1) = input->at(s).t();
      }

      const arma::fmat &xy = x_stages.submat(0, 0, x_stages.n_rows - 1, 1);
      const arma::fmat &wh = x_stages.submat(0, 2, x_stages.n_rows - 1, 3);

      x_stages.submat(0, 0, x_stages.n_rows - 1, 1) = (xy * 2 + grids_[stage]) * strides_[stage];
      x_stages.submat(0, 2, x_stages.n_rows - 1, 3) = arma::pow((wh * 2), 2) % anchor_grids_[stage];
    }
    concat_rows += x_stages_tensor->rows();
    zs.at(stage) = x_stages_tensor;
  }

  uint32_t current_rows = 0;
  arma::fcube f1(concat_rows, classes_info, batch_size);
  for (const auto &z : zs) {
    f1.subcube(current_rows, 0, 0, current_rows + z->rows() - 1, classes_info - 1, batch_size - 1) = z->data();
    current_rows += z->rows();
  }

  for (int i = 0; i < f1.n_slices; ++i) {
    auto &output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(1, concat_rows, classes_info);
    }
    output->at(0) = f1.slice(i);
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus YoloDetectLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                      std::shared_ptr<Layer> &yolo_detect_layer) {

  CHECK(op != nullptr) << "Yolo detect operator is nullptr";

  const auto &attrs = op->attribute;
  CHECK(!attrs.empty()) << "Operator attributes is empty!";
  if (attrs.find("pnnx_5") == attrs.end()) {
    LOG(ERROR) << "Can not find the in yolo strides attribute";
    return ParseParameterAttrStatus::kAttrMissingYoloStrides;
  }

  const auto &stages_attr = attrs.at("pnnx_5");
  if (stages_attr->shape.empty()) {
    LOG(ERROR) << "Can not find the in yolo strides attribute";
    return ParseParameterAttrStatus::kAttrMissingYoloStrides;
  }

  int stages_number = stages_attr->shape.at(0);
  CHECK(stages_number == 3) << "Only support three stages yolo detect head";
  const std::vector<float> &strides = stages_attr->get<float>();
  CHECK(strides.size() == stages_number) << "Stride number is not equal to strides";

  std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers(stages_number);
  int32_t num_classes = -1;
  for (int i = 0; i < stages_number; ++i) {
    const std::string &weight_name = "m." + std::to_string(i) + ".weight";
    if (attrs.find(weight_name) == attrs.end()) {
      LOG(ERROR) << "Can not find the in weight attribute " << weight_name;
      return ParseParameterAttrStatus::kAttrMissingWeight;
    }

    const auto &conv_attr = attrs.at(weight_name);
    const auto &out_shapes = conv_attr->shape;
    CHECK(out_shapes.size() == 4);

    const int out_channels = out_shapes.at(0);
    if (num_classes == -1) {
      CHECK(out_channels % stages_number == 0)
              << "The number of output channel is wrong, it should divisible by stages number";
      num_classes = out_channels / stages_number - 5;
      CHECK(num_classes > 0) << "The number of object classes must greater than zero";
    }
    const int in_channels = out_shapes.at(1);
    const int kernel_h = out_shapes.at(2);
    const int kernel_w = out_shapes.at(3);
    conv_layers.at(i) =
        std::make_shared<ConvolutionLayer>(out_channels, in_channels, kernel_h, kernel_w, 0, 0, 1, 1, 1);
    const std::vector<float> &weights = conv_attr->get<float>();
    conv_layers.at(i)->set_weights(weights);

    const std::string &bias_name = "m." + std::to_string(i) + ".bias";
    if (attrs.find(bias_name) == attrs.end()) {
      LOG(ERROR) << "Can not find the in bias attribute";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }
    const auto &bias_attr = attrs.at(bias_name);
    const std::vector<float> &bias = bias_attr->get<float>();
    conv_layers.at(i)->set_bias(bias);
  }

  std::vector<arma::fmat> anchor_grids;
  for (int i = 4; i >= 0; i -= 2) {
    const std::string &pnnx_name = "pnnx_" + std::to_string(i);
    const auto &anchor_grid_attr = attrs.find(pnnx_name);
    if (anchor_grid_attr == attrs.end()) {
      LOG(ERROR) << "Can not find the in yolo anchor grides attribute";
      return ParseParameterAttrStatus::kAttrMissingYoloAnchorGrides;
    }
    const auto &anchor_grid = anchor_grid_attr->second;
    const auto &anchor_shapes = anchor_grid->shape;
    const std::vector<float> &anchor_weight_data = anchor_grid->get<float>();
    CHECK(!anchor_shapes.empty() && anchor_shapes.size() == 5 && anchor_shapes.front() == 1);

    const uint32_t anchor_rows = anchor_shapes.at(1) * anchor_shapes.at(2) * anchor_shapes.at(3);
    const uint32_t anchor_cols = anchor_shapes.at(4);
    CHECK(anchor_weight_data.size() == anchor_cols * anchor_rows);

    arma::fmat anchor_grid_matrix(anchor_weight_data.data(), anchor_cols, anchor_rows);
    anchor_grids.emplace_back(anchor_grid_matrix.t());
  }

  std::vector<arma::fmat> grids;
  std::vector<int32_t> grid_indexes{6, 3, 1};
  for (const auto grid_index : grid_indexes) {
    const std::string &pnnx_name = "pnnx_" + std::to_string(grid_index);
    const auto &grid_attr = attrs.find(pnnx_name);
    if (grid_attr == attrs.end()) {
      LOG(ERROR) << "Can not find the in yolo grides attribute";
      return ParseParameterAttrStatus::kAttrMissingYoloGrides;
    }
    const auto &grid = grid_attr->second;
    const auto &shapes = grid->shape;
    const std::vector<float> &weight_data = grid->get<float>();
    CHECK(!shapes.empty() && shapes.size() == 5 && shapes.front() == 1);
    const uint32_t grid_rows = shapes.at(1) * shapes.at(2) * shapes.at(3);
    const uint32_t grid_cols = shapes.at(4);
    CHECK(weight_data.size() == grid_cols * grid_rows);

    arma::fmat matrix(weight_data.data(), grid_cols, grid_rows);
    grids.emplace_back(matrix.t());
  }
  yolo_detect_layer =
      std::make_shared<YoloDetectLayer>(stages_number, num_classes, strides, anchor_grids, grids, conv_layers);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kYoloGetInstance("models.yolo.Detect", YoloDetectLayer::GetInstance);

}