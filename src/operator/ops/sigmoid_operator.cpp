#include "operator/operators.hpp"
#include <iostream>
#include <cmath>

// implement inferOutputShapes and inferOutputDataTypes
std::vector<std::vector<size_t>> SigmoidOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() != 1)
    {
        return {};
    }

    return {inputs.at(0).getDims()};
}

std::vector<TensorDataType> SigmoidOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                  const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {};
    }
    return {inputs.at(0).getDataType()};
}

OperatorExecuteResult SigmoidOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                               const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
{
    DeviceType deviceType = device->getType();

    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::SigmoidOperatorImpl::execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::SigmoidOperatorImpl::execute(inputs, outputs, attributes, device);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
