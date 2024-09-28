#include <iostream>
#include <type_traits>

#include "device/device.hpp"
#include "operator/operators.hpp"
#include "operator/elementwise_operator.hpp"

std::vector<std::vector<size_t>> AddOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return inferElementwiseOutputShapes(inputs);
}

std::vector<TensorDataType> AddOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                              const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return inferElementwiseOutputDataTypes(inputs);
}

OperatorExecuteResult AddOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes,
                                           Device *device)
{

    if (inputs.size() != 2)
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    if (outputs.size() != 1)
    {
        return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
    }

    const TensorDataType dataType = inputs[0].getDataType();
    for (size_t i = 1; i < inputs.size(); i++)
    {
        if (inputs[i].getDataType() != dataType)
        {
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }

    if (outputs[0]->getDims().empty())
    {
        return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
    }

    if (outputs.empty() || outputs[0] == nullptr)
    {
        return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
    }

    DeviceType deviceType = device->getType();

    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::AddOperatorImpl::execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::AddOperatorImpl::execute(inputs, outputs, attributes, device);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
