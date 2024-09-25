#include "operator/operators.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

std::vector<std::vector<size_t>> ConvOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                 const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    const Tensor &X = inputs.at(0);
    const Tensor &W = inputs.at(1);

    // Get attributes with defaults
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    int64_t group = 1;

    for (const auto &[key, value] : attributes)
    {
        if (key == "pads")
            pads = std::get<std::vector<int64_t>>(value);
        else if (key == "strides")
            strides = std::get<std::vector<int64_t>>(value);
        else if (key == "dilations")
            dilations = std::get<std::vector<int64_t>>(value);
        else if (key == "group")
            group = std::get<int64_t>(value);
    }
    const std::vector<size_t> &X_dims = X.getDims(); // (N, C, H, W)
    const std::vector<size_t> &W_dims = W.getDims(); // (M, C/group, kH, kW)

    // Check dimensions
    if (X_dims.size() != 4 || W_dims.size() != 4)
    {
        return {};
    }

    size_t N = X_dims[0];
    size_t C = X_dims[1];
    size_t H = X_dims[2];
    size_t W_in = X_dims[3];
    size_t M = W_dims[0];
    size_t kH = W_dims[2];
    size_t kW = W_dims[3];

    // Validate kernel shape
    if (kH > H + pads[0] + pads[2] || kW > W_in + pads[1] + pads[3])
    {
        return {}; // Shape mismatch
    }

    // Calculate output dimensions
    size_t H_out = static_cast<size_t>(std::floor((H + pads[0] + pads[2] - kH * dilations[0] + strides[0]) / strides[0]));
    size_t W_out = static_cast<size_t>(std::floor((W_in + pads[1] + pads[3] - kW * dilations[1] + strides[1]) / strides[1]));

    return {{N, M, H_out, W_out}};
}

std::vector<TensorDataType> ConvOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                               const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {TensorDataType::UNDEFINED};
    }

    return {inputs.at(0).getDataType()};
}

OperatorExecuteResult ConvOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                            const std::unordered_map<std::string, Node::AttributeValue> &attributes, DeviceType deviceType)
{
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::ConvOperatorImpl().execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::ConvOperatorImpl().execute(inputs, outputs, attributes);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
