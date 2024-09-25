#ifndef OPERATOR_IMPL_HPP
#define OPERATOR_IMPL_HPP

#include "operator/operator.hpp"

class OperatorImpl
{
public:
    // DeviceType getDeviceType() const
    // {
    //     return deviceType;
    // }
    static OperatorExecuteResult execute(const std::vector<Tensor> &inputs,
                                         std::vector<Tensor *> &outputs,
                                         const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        return OperatorExecuteResult::UNSUPPORTED_OPERATION;
    }

private:
    // DeviceType deviceType;
};

#endif // OPERATOR_IMPL_HPP
