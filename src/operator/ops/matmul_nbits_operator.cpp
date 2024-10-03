#include "operator/operators.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdint>

std::vector<std::vector<size_t>> MatMulNBitsOperator::inferOutputShapes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() < 3)
    {
        return {};
    }
    const size_t batch_size = inputs.at(0).getDims().at(0);
    const size_t M = inputs.at(0).getDims().at(1);
    const size_t N = inputs.at(1).getDims().at(0);

    return {{batch_size, M, N}};
}

std::vector<TensorDataType> MatMulNBitsOperator::inferOutputDataTypes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {TensorDataType::UNDEFINED};
    }

    return {inputs.at(0).getDataType()};
}

OperatorExecuteResult MatMulNBitsOperator::execute(
    const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
{
    if (inputs.size() < 3)
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }
    if (outputs.empty() || outputs[0] == nullptr)
    {
        return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
    }

    DeviceType deviceType = device->getType();
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::MatMulNBitsOperatorImpl::execute(inputs, outputs, attributes);

#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::MatMulNBitsOperatorImpl::execute(inputs, outputs, attributes, device);
#endif

    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
