#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "tensor/tensor.hpp"
#include "graph/node.hpp"
#include "device/device.hpp"

enum class OperatorType
{
    Add,
    Mul,
    Sub,
    Div,
    MatMul,
    Conv,
    Sigmoid,
    Constant,
    Split,
    Concat,
    Slice,
    Gather,
    Shape,
    Reshape,
    Softmax,
    Transpose,
    Resize,
    MaxPool
};

enum class OperatorExecuteResult
{
    SUCCESS,
    INPUT_TENSOR_ERROR,
    INPUT_TENSOR_VALUE_ERROR,
    OUTPUT_TENSOR_ERROR,
    ATTRIBUTE_ERROR,
    DATA_TYPE_ERROR,
    SHAPE_MISMATCH_ERROR,
    UNSUPPORTED_OPERATION,
    MEMORY_ALLOCATION_ERROR,
    UNKNOWN_ERROR,
    DEVICE_UNSUPPORTED
};

namespace OperatorUtils
{
    std::string OperatorExecuteResultToString(OperatorExecuteResult result);
    std::string OperatorTypeToString(OperatorType type);
}

class Operator
{
public:
    Operator(OperatorType type) : type(type) {}

    virtual ~Operator() = default;

    virtual OperatorExecuteResult execute(const std::vector<Tensor> &inputs,
                                          std::vector<Tensor *> &outputs,
                                          const std::unordered_map<std::string, Node::AttributeValue> &attributes, DeviceType deviceType = DeviceType::CPU)
    {
        return OperatorExecuteResult::UNSUPPORTED_OPERATION;
    }

    virtual std::vector<std::vector<size_t>> inferOutputShapes(const std::vector<Tensor> &inputs,
                                                               const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        return {{}};
    }

    virtual std::vector<TensorDataType> inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                             const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        if (inputs.empty())
        {
            return {TensorDataType::UNDEFINED};
        }
        return {inputs[0].getDataType()};
    }

    OperatorType getType() const { return type; }

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
