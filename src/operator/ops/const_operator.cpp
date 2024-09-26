#include "operator/operators.hpp"
#include <iostream>

std::vector<std::vector<size_t>> ConstantOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (attributes.count("value"))
    {
        return {std::get<Tensor>(attributes.at("value")).getDims()};
    }
    return {};
}

std::vector<TensorDataType> ConstantOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (attributes.count("value"))
    {
        return {std::get<Tensor>(attributes.at("value")).getDataType()};
    }
    return {};
}

OperatorExecuteResult ConstantOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes, DeviceType deviceType)
{
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::ConstantOperatorImpl().execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::ConstantOperatorImpl().execute(inputs, outputs, attributes);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}