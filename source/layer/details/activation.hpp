//
// Created by fss on 23-12-14.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
#include "data/tensor.hpp"
#include "layer/abstract/non_param_layer.hpp"
#include "status_code.hpp"
namespace kuiper_infer {
namespace activation {
using ActivationFunc = std::function<void(sftensor, sftensor)>;

enum class ActivationType {
  kActivatetionUnknown = -1,
  kActivationRelu = 0,
  kActivationSilu = 1,
  kActivationSigmoid = 2,
  kActivationHardSwish = 3,
  kActivationHardSigmoid = 4,
  kActivationRelu6 = 5,
};

std::string ActivationTypeToString(ActivationType type);

class ActivationLayer : public NonParamLayer {
 public:
  explicit ActivationLayer(activation::ActivationType type, std::string layer_name);

  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

 private:
  ActivationType act_type_ = ActivationType::kActivatetionUnknown;
};
}  // namespace activation
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
