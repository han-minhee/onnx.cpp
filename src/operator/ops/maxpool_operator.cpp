#include "operator/operators.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

std::vector<std::vector<size_t>> MaxPoolOperator::inferOutputShapes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    const Tensor &X = inputs.at(0);
    const std::vector<size_t> &X_dims = X.getDims();

    if (X_dims.size() < 4)
    {
        return {};
    }

    size_t spatial_rank = X_dims.size() - 2;

    // Initialize vectors with default values
    std::vector<int64_t> kernel_shape(spatial_rank, 1);
    std::vector<int64_t> pads(spatial_rank * 2, 0);
    std::vector<int64_t> strides(spatial_rank, 1);
    std::vector<int64_t> dilations(spatial_rank, 1);
    std::string auto_pad = "NOTSET";
    int64_t ceil_mode = 0;

    // Extract attribute values and resize vectors accordingly
    for (const auto &[key, value] : attributes)
    {
        if (key == "kernel_shape")
        {
            kernel_shape = std::get<std::vector<int64_t>>(value);
            if (kernel_shape.size() != spatial_rank)
            {
                kernel_shape.resize(spatial_rank, 1); // Ensure correct size
            }
        }
        else if (key == "pads")
        {
            pads = std::get<std::vector<int64_t>>(value);
            if (pads.size() != spatial_rank * 2)
            {
                pads.resize(spatial_rank * 2, 0); // Ensure correct size
            }
        }
        else if (key == "strides")
        {
            strides = std::get<std::vector<int64_t>>(value);
            if (strides.size() != spatial_rank)
            {
                strides.resize(spatial_rank, 1); // Ensure correct size
            }
        }
        else if (key == "dilations")
        {
            dilations = std::get<std::vector<int64_t>>(value);
            if (dilations.size() != spatial_rank)
            {
                dilations.resize(spatial_rank, 1); // Ensure correct size
            }
        }
        else if (key == "auto_pad")
        {
            auto_pad = std::get<std::string>(value);
        }
        else if (key == "ceil_mode")
        {
            ceil_mode = std::get<int64_t>(value);
        }
    }

    std::vector<size_t> output_dims = {X_dims[0], X_dims[1]};

    for (size_t i = 0; i < spatial_rank; ++i)
    {
        int64_t input_size = X_dims[i + 2];
        int64_t kernel = kernel_shape[i];
        int64_t pad_total = pads[i] + pads[i + spatial_rank];
        int64_t dilation = dilations[i];
        int64_t stride = strides[i];

        int64_t output_size = 0;

        if (auto_pad == "VALID")
        {
            output_size = (input_size - dilation * (kernel - 1) - 1) / stride + 1;
        }
        else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER")
        {
            output_size = static_cast<int64_t>(std::ceil(input_size / static_cast<double>(stride)));
        }
        else
        {
            int64_t numerator = input_size + pad_total - dilation * (kernel - 1) - 1;
            if (ceil_mode)
            {
                output_size = static_cast<int64_t>(std::ceil(numerator / static_cast<double>(stride))) + 1;
            }
            else
            {
                output_size = static_cast<int64_t>(std::floor(numerator / static_cast<double>(stride))) + 1;
            }
        }

        output_dims.push_back(static_cast<size_t>(std::max<int64_t>(output_size, 1)));
    }

    return {output_dims};
}

std::vector<TensorDataType> MaxPoolOperator::inferOutputDataTypes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {TensorDataType::UNDEFINED};
    }

    return {inputs.at(0).getDataType()};
}

OperatorExecuteResult MaxPoolOperator::execute(
    const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
{

    if (inputs.size() < 1 || outputs.empty() || outputs[0] == nullptr)
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    DeviceType deviceType = device->getType();
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::MaxPoolOperatorImpl::execute(inputs, outputs, attributes);

#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::MaxPoolOperatorImpl::execute(inputs, outputs, attributes, device);
#endif

    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
