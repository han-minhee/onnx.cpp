#include <functional>
#include "operator/operators.hpp"
#include "operator/elementwise_operator.hpp"

std::vector<std::vector<size_t>> DivOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return inferElementwiseOutputShapes(inputs);
}

std::vector<TensorDataType> DivOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                              const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return inferElementwiseOutputDataTypes(inputs);
}

/// FIXME: Implement division by zero check
OperatorExecuteResult DivOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device &device)
{
    DeviceType deviceType = device.getType();

    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::DivOperatorImpl::execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::DivOperatorImpl::execute(inputs, outputs, attributes, device);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
