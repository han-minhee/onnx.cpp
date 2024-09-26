#include "operator/operators.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

std::vector<std::vector<size_t>> ResizeOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    // Validate the number of inputs
    if (inputs.size() < 1 || inputs.size() > 4)
    {
        throw std::invalid_argument("Invalid number of input tensors for ResizeOperator.");
    }

    const Tensor &input_tensor = inputs[0];
    std::vector<size_t> input_shape = input_tensor.getDims();
    std::vector<size_t> output_shape;
    std::vector<float> scales;
    std::vector<int64_t> sizes;
    bool has_scales = false;
    bool has_sizes = false;

    if (inputs.size() >= 3)
    {
        const Tensor &scales_tensor = inputs[2];
        if (scales_tensor.getDataType() != TensorDataType::FLOAT32)
        {
            throw std::invalid_argument("Scales tensor must be of FLOAT32 data type.");
        }

        scales = std::vector<float>(scales_tensor.data<float>(), scales_tensor.data<float>() + scales_tensor.getNumElements());
        has_scales = true;
    }

    if (inputs.size() == 4)
    {
        const Tensor &sizes_tensor = inputs[3];

        if (sizes_tensor.getDataType() != TensorDataType::INT64)
        {
            throw std::invalid_argument("Sizes tensor must be of INT64 data type.");
        }

        sizes = std::vector<int64_t>(sizes_tensor.data<int64_t>(), sizes_tensor.data<int64_t>() + sizes_tensor.getNumElements());
        has_sizes = true;
    }

    if (has_scales == has_sizes)
    {
        throw std::invalid_argument("Exactly one of 'scales' or 'sizes' must be specified.");
    }

    if (has_sizes)
    {
        output_shape = std::vector<size_t>(sizes.begin(), sizes.end());
    }
    else if (has_scales)
    {
        output_shape.resize(input_shape.size());
        for (size_t i = 0; i < input_shape.size(); ++i)
        {
            output_shape[i] = static_cast<size_t>(std::floor(input_shape[i] * scales[i]));
        }
    }

    return {output_shape};
}

std::vector<TensorDataType> ResizeOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                 const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        throw std::invalid_argument("No input tensors provided.");
    }
    return {inputs[0].getDataType()};
}

OperatorExecuteResult ResizeOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                              const std::unordered_map<std::string, Node::AttributeValue> &attributes, DeviceType deviceType)
{
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::ResizeOperatorImpl::execute(inputs, outputs, attributes);

#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::ResizeOperatorImpl::execute(inputs, outputs, attributes);

#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
