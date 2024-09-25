#include <functional>
#include "operator/operators.hpp"
#include "operator/aux_operator/elementwise_operator.hpp"

std::vector<std::vector<size_t>> MulOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return inferElementwiseOutputShapes(inputs);
}

std::vector<TensorDataType> MulOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                              const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return inferElementwiseOutputDataTypes(inputs);
}

OperatorExecuteResult MulOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes, DeviceType deviceType)
{
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::MulOperatorImpl::execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::MulOperatorImpl::execute(inputs, outputs, attributes);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
