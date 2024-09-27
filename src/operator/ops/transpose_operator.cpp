#include "operator/operators.hpp"
#include <iostream>
// implement inferOutputShapes and inferOutputDataTypes
std::vector<std::vector<size_t>> TransposeOperator::inferOutputShapes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("TransposeOperator requires exactly one input tensor.");
    }

    const Tensor &input_tensor = inputs[0];
    const std::vector<size_t> &input_shape = input_tensor.getDims();
    size_t rank = input_shape.size();

    // Determine the permutation order
    std::vector<size_t> perm(rank);
    if (attributes.find("perm") != attributes.end())
    {
        const std::vector<int64_t> &perm_attr = std::get<std::vector<int64_t>>(attributes.at("perm"));

        // Ensure the length of perm matches the rank of the input tensor
        if (perm_attr.size() != rank)
        {
            throw std::invalid_argument("The 'perm' attribute length must match the rank of the input tensor.");
        }

        // Convert perm_attr to perm
        for (size_t i = 0; i < rank; ++i)
        {
            perm[i] = static_cast<size_t>(perm_attr[i]);
        }
    }
    else
    {
        // Default permutation: reverse the dimensions
        for (size_t i = 0; i < rank; ++i)
        {
            perm[i] = rank - 1 - i;
        }
    }

    // Compute the output shape based on the perm attribute
    std::vector<size_t> output_shape(rank);
    for (size_t i = 0; i < rank; ++i)
    {
        output_shape[i] = input_shape[perm[i]];
    }

    return {output_shape};
}

std::vector<TensorDataType> TransposeOperator::inferOutputDataTypes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("TransposeOperator requires exactly one input tensor.");
    }

    // The output data type remains the same as the input tensor data type
    return {inputs[0].getDataType()};
}

OperatorExecuteResult TransposeOperator::execute(
    const std::vector<Tensor> &inputs,
    std::vector<Tensor *> &outputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes,
    Device &device)
{

    DeviceType deviceType = device.getType();
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::TransposeOperatorImpl::execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::TransposeOperatorImpl::execute(inputs, outputs, attributes, device);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
