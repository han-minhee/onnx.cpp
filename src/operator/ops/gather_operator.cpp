#include "operator/operators.hpp"
#include <iostream>
#include <type_traits>

// Implement inferOutputShapes and inferOutputDataTypes
std::vector<std::vector<size_t>> GatherOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    // Check for the required inputs
    if (inputs.size() != 2)
    {
        return {};
    }

    const Tensor &data = inputs[0];
    const Tensor &indices = inputs[1];

    // Get the shapes of input tensors
    const std::vector<size_t> &data_shape = data.getDims();
    const std::vector<size_t> &indices_shape = indices.getDims();
    size_t data_rank = data_shape.size();
    size_t indices_rank = indices_shape.size();

    // Get the axis attribute (default is 0)
    int64_t axis = 0;
    if (attributes.find("axis") != attributes.end())
    {
        axis = std::get<int64_t>(attributes.at("axis"));
    }

    // Adjust axis for negative values
    if (axis < 0)
    {
        axis += data_rank;
    }

    if (axis < 0 || static_cast<size_t>(axis) >= data_rank)
    {
        return {};
    }

    // Determine the shape of the output tensor
    std::vector<size_t> output_shape;
    output_shape.reserve(indices_rank + data_rank - 1);

    // Copy the shape of indices into output shape
    output_shape.insert(output_shape.end(), indices_shape.begin(), indices_shape.end());

    // Copy the remaining shape from data excluding the specified axis
    for (size_t i = 0; i < data_rank; ++i)
    {
        if (i != static_cast<size_t>(axis))
        {
            output_shape.push_back(data_shape[i]);
        }
    }

    return {output_shape};
}

std::vector<TensorDataType> GatherOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                 const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() != 2)
    {
        return {};
    }

    const TensorDataType dataType = inputs[0].getDataType();
    if (inputs[1].getDataType() != TensorDataType::INT64)
    {
        return {};
    }

    return {dataType};
}

OperatorExecuteResult GatherOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                              const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device& device)
{
    DeviceType deviceType = device.getType(); 
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::GatherOperatorImpl::execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::GatherOperatorImpl::execute(inputs, outputs, attributes, device);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
