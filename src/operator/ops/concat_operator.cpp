#include "operator/operators.hpp"
#include <iostream>
#include <algorithm>

// Modify the inferOutputShapes function
std::vector<std::vector<size_t>> ConcatOperator::inferOutputShapes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {};
    }

    // Get 'axis' attribute
    if (attributes.find("axis") == attributes.end())
    {
        return {}; // axis is required
    }

    int64_t axis = std::get<int64_t>(attributes.at("axis"));
    size_t rank = inputs[0].getNDim();

    // Adjust negative axis
    if (axis < 0)
    {
        axis += static_cast<int64_t>(rank);
    }

    if (axis < 0 || axis >= static_cast<int64_t>(rank))
    {
        return {}; // Invalid axis
    }

    // Initialize output shape with the shape of the first input
    std::vector<size_t> output_shape = inputs[0].getDims();

    // Sum the sizes along the concatenation axis
    size_t axis_size = output_shape[axis];

    for (size_t i = 1; i < inputs.size(); ++i)
    {
        const auto &input_shape = inputs[i].getDims();

        // Check that all inputs have the same rank
        if (input_shape.size() != rank)
        {
            return {}; // Shape mismatch
        }

        // Check that dimensions match except along the concatenation axis
        for (size_t dim = 0; dim < rank; ++dim)
        {
            if (dim == static_cast<size_t>(axis))
            {
                continue; // Skip concatenation axis
            }
            if (input_shape[dim] != output_shape[dim])
            {
                return {}; // Return empty to indicate a shape mismatch
            }
        }

        // Accumulate size along the axis
        axis_size += input_shape[axis];
    }

    // Set the concatenated size along the axis
    output_shape[axis] = axis_size;

    return {output_shape};
}

std::vector<TensorDataType> ConcatOperator::inferOutputDataTypes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {TensorDataType::UNDEFINED};
    }

    // Check that all inputs have the same data type
    TensorDataType data_type = inputs[0].getDataType();
    for (const auto &input : inputs)
    {
        if (input.getDataType() != data_type)
        {
            return {TensorDataType::UNDEFINED};
        }
    }

    return {data_type};
}

OperatorExecuteResult ConcatOperator::execute(
    const std::vector<Tensor> &inputs,
    std::vector<Tensor *> &outputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device& device)
{
    
    DeviceType deviceType = device.getType();

    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::ConcatOperatorImpl::execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::ConcatOperatorImpl::execute(inputs, outputs, attributes, device);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
