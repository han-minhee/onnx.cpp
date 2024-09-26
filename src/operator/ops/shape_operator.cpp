#include "operator/operators.hpp"
#include <iostream>

// Implement inferOutputShapes and inferOutputDataTypes
std::vector<std::vector<size_t>> ShapeOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                  const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    // Validate the number of inputs
    if (inputs.size() != 1)
    {
        return {};
    }

    const Tensor &input_tensor = inputs[0];
    const std::vector<size_t> &input_shape = input_tensor.getDims();
    size_t rank = input_shape.size();

    // Get the 'start' attribute, default is 0
    int64_t start = 0;
    if (attributes.find("start") != attributes.end())
    {
        start = std::get<int64_t>(attributes.at("start"));
    }

    // Get the 'end' attribute, default is rank (all dimensions included)
    int64_t end = static_cast<int64_t>(rank);
    if (attributes.find("end") != attributes.end())
    {
        end = std::get<int64_t>(attributes.at("end"));
    }

    // Adjust negative indices
    if (start < 0)
    {
        start += static_cast<int64_t>(rank);
    }
    if (end < 0)
    {
        end += static_cast<int64_t>(rank);
    }

    // Clamp start and end to be within the valid range [0, rank]
    start = static_cast<int64_t>(std::max(0, std::min(static_cast<int>(start), static_cast<int>(rank))));
    end = static_cast<int64_t>(std::max(0, std::min(static_cast<int>(end), static_cast<int>(rank))));

    // Compute the output shape slice
    std::vector<size_t> output_shape_slice;
    for (int i = start; i < end; ++i)
    {
        output_shape_slice.push_back(static_cast<int64_t>(input_shape[i]));
    }

    return {output_shape_slice};
}

std::vector<TensorDataType> ShapeOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return {TensorDataType::INT64};
}

OperatorExecuteResult ShapeOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                             const std::unordered_map<std::string, Node::AttributeValue> &attributes, DeviceType deviceType)
{
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::ShapeOperatorImpl::execute(inputs, outputs, attributes);

#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::ShapeOperatorImpl::execute(inputs, outputs, attributes);

#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
