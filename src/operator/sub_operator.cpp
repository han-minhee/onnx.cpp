#include <iostream>
#include <type_traits>

#include "operator/operators.hpp"
#include "operator/aux_operator/elementwise_operator.hpp"

std::vector<std::vector<size_t>> SubOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return inferElementwiseOutputShapes(inputs);
}

std::vector<TensorDataType> SubOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                              const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return inferElementwiseOutputDataTypes(inputs);
}

OperatorExecuteResult SubOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes,
                                           DeviceType deviceType = DeviceType::CPU)
{
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::SubOperatorImpl().execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::SubOperatorImpl().execute(inputs, outputs, attributes);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}