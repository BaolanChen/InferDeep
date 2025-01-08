//
// Created by fss on 23-12-14.
//
#include "activation.hpp"
#include "simd.hpp"
namespace kuiper_infer {
namespace activation {
std::string ActivationTypeToString(ActivationType type) {
  std::string activate_type;
  switch (type) {
    case ActivationType::kActivationRelu: {
      activate_type = "Relu";
      break;
    }
    case ActivationType::kActivationSilu: {
      activate_type = "Silu";
      break;
    }
    case ActivationType::kActivationRelu6: {
      activate_type = "Relu6";
      break;
    }
    case ActivationType::kActivationSigmoid: {
      activate_type = "Sigmoid";
      break;
    }
    case ActivationType::kActivationHardSigmoid: {
      activate_type = "HardSigmoid";
      break;
    }
    case ActivationType::kActivationHardSwish: {
      activate_type = "HardSwish";
      break;
    }
    default: {
      activate_type = "Unknown";
      break;
    }
  }
  return activate_type;
}

StatusCode ActivationLayer::Check(const std::vector<sftensor>& inputs,
                                  const std::vector<sftensor>& outputs) {
  const std::string& activation_type = ActivationTypeToString(act_type_);
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the " + activation_type + " layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the " + activation_type + " layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the " + activation_type +
                      " layer do not match";
    return StatusCode::kInferDimMismatch;
  }
  return StatusCode::kSuccess;
}

StatusCode ActivationLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  StatusCode check_status = Check(inputs, outputs);
  if (check_status != StatusCode::kSuccess) {
    return check_status;
  }

  const uint32_t batch_size = inputs.size();
  const std::string& act_type_str = ActivationTypeToString(act_type_);
  ActivationFunc activation_function = ApplySSEActivation(act_type_);
#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the " + act_type_str + " layer has an empty tensor " << i
        << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }
    CHECK(output != nullptr && output->shapes() == input->shapes())
        << "The input and output tensor shapes of the " + act_type_str + " layer do not match " << i
        << " th";
    activation_function(input, output);
  }
  return StatusCode::kSuccess;
}

ActivationLayer::ActivationLayer(activation::ActivationType type, std::string layer_name)
    : NonParamLayer(std::move(layer_name)), act_type_(type) {}
}  // namespace activation
}  // namespace kuiper_infer