#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "graph/node.hpp"
#include "tensor/tensor.hpp"
#include "device/device.hpp"
#include "enums.hpp"

namespace OperatorUtils
{
    std::string OperatorExecuteResultToString(OperatorExecuteResult result);
    std::string OperatorTypeToString(OperatorType type);
}

class Operator
{
public:
    /// FIXME: type should also be static
    Operator(OperatorType type) : type(type) {}

    virtual ~Operator() = default;

    static OperatorExecuteResult execute(const std::vector<Tensor> &inputs,
                                         std::vector<Tensor *> &outputs,
                                         const std::unordered_map<std::string, Node::AttributeValue> &attributes, DeviceType deviceType = DeviceType::CPU)
    {
        return OperatorExecuteResult::UNSUPPORTED_OPERATION;
    }

    static std::vector<std::vector<size_t>> inferOutputShapes(const std::vector<Tensor> &inputs,
                                                              const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        return {{}};
    }

    static std::vector<TensorDataType> inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                            const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        if (inputs.empty())
        {
            return {TensorDataType::UNDEFINED};
        }
        return {inputs[0].getDataType()};
    }

    OperatorType getType() { return type; }

private:
    OperatorType type;
};

class OperatorFactory
{
public:
    static std::unique_ptr<Operator> createOperator(const std::string &op_type);

private:
    OperatorFactory() = default;
};

/// FIXME: These auxiliary functions should be moved to a different file
std::vector<size_t> compute_broadcast_shape(const std::vector<std::vector<size_t>> &shapes);
std::vector<size_t> compute_broadcast_strides(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape);

#endif // OPERATOR_HPP
