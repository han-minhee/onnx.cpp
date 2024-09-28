#include <stdexcept>
#include <iostream>

#include "operator/operator.hpp"
#include "operator/operators.hpp"

std::vector<size_t> compute_broadcast_shape(const std::vector<std::vector<size_t>> &shapes)
{
    std::vector<size_t> result_shape;
    size_t max_rank = 0;

    for (const auto &shape : shapes)
    {
        if (shape.size() > max_rank)
        {
            max_rank = shape.size();
        }
    }

    result_shape.resize(max_rank, 1);

    for (size_t i = 0; i < max_rank; ++i)
    {
        size_t dim = 1;
        for (const auto &shape : shapes)
        {
            size_t idx = shape.size() >= max_rank - i ? shape[shape.size() - (max_rank - i)] : 1;
            if (idx != 1 && dim != 1 && idx != dim)
            {

                return {};
            }
            if (idx > dim)
            {
                dim = idx;
            }
        }
        result_shape[i] = dim;
    }

    return result_shape;
}

std::vector<size_t> compute_broadcast_strides(const std::vector<size_t> &input_shape,
                                              const std::vector<size_t> &output_shape)
{
    size_t input_rank = input_shape.size();
    size_t output_rank = output_shape.size();

    std::vector<size_t> adjusted_shape = input_shape;
    if (input_rank < output_rank)
    {
        adjusted_shape.insert(adjusted_shape.begin(), output_rank - input_rank, 1);
    }

    std::vector<size_t> strides(output_rank, 1);
    for (int i = output_rank - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * adjusted_shape[i + 1];
    }

    return strides;
}

namespace OperatorUtils
{
    std::string OperatorExecuteResultToString(OperatorExecuteResult result)
    {
        switch (result)
        {
        case OperatorExecuteResult::SUCCESS:
            return "SUCCESS";
        case OperatorExecuteResult::INPUT_TENSOR_ERROR:
            return "INPUT_TENSOR_ERROR";
        case OperatorExecuteResult::INPUT_TENSOR_VALUE_ERROR:
            return "INPUT_TENSOR_VALUE_ERROR";
        // case OperatorExecuteResult::DIVIDE_BY_ZERO_ERROR:
        //     return "DIVIDE_BY_ZERO_ERROR";
        case OperatorExecuteResult::OUTPUT_TENSOR_ERROR:
            return "OUTPUT_TENSOR_ERROR";
        case OperatorExecuteResult::ATTRIBUTE_ERROR:
            return "ATTRIBUTE_ERROR";
        case OperatorExecuteResult::DATA_TYPE_ERROR:
            return "DATA_TYPE_ERROR";
        case OperatorExecuteResult::SHAPE_MISMATCH_ERROR:
            return "SHAPE_MISMATCH_ERROR";
        case OperatorExecuteResult::UNSUPPORTED_OPERATION:
            return "UNSUPPORTED_OPERATION";
        case OperatorExecuteResult::MEMORY_ALLOCATION_ERROR:
            return "MEMORY_ALLOCATION_ERROR";
        case OperatorExecuteResult::DEVICE_UNSUPPORTED:
            return "DEVICE_UNSUPPORTED";
        case OperatorExecuteResult::UNKNOWN_ERROR:
            return "UNKNOWN_ERROR";
        case OperatorExecuteResult::NOT_IMPLEMENTED:
            return "NOT_IMPLEMENTED";
        default:
            return "UNKNOWN";
        }
    }

#define OPERATOR_TYPE_TO_STRING(type) \
    case OperatorType::type:          \
        return #type;

    std::string OperatorTypeToString(OperatorType type)
    {
        switch (type)
        {
            OPERATOR_TYPE_TO_STRING(Add)
            OPERATOR_TYPE_TO_STRING(Conv)
            OPERATOR_TYPE_TO_STRING(Constant)
            OPERATOR_TYPE_TO_STRING(Sub)
            OPERATOR_TYPE_TO_STRING(Reshape)
            OPERATOR_TYPE_TO_STRING(Split)
            OPERATOR_TYPE_TO_STRING(Concat)
            OPERATOR_TYPE_TO_STRING(MatMul)
            OPERATOR_TYPE_TO_STRING(Div)
            OPERATOR_TYPE_TO_STRING(Mul)
            OPERATOR_TYPE_TO_STRING(Sigmoid)
            OPERATOR_TYPE_TO_STRING(Slice)
            OPERATOR_TYPE_TO_STRING(Gather)
            OPERATOR_TYPE_TO_STRING(Shape)
            OPERATOR_TYPE_TO_STRING(Softmax)
            OPERATOR_TYPE_TO_STRING(Transpose)
            OPERATOR_TYPE_TO_STRING(Resize)
            OPERATOR_TYPE_TO_STRING(MaxPool)
        default:
            return "Unknown";
        }
    }

#undef OPERATOR_TYPE_TO_STRING
#define OPERATOR_TYPE_FROM_STRING(baseName) \
    if (operatorName == #baseName)          \
    {                                       \
        return OperatorType::baseName;      \
    }

    OperatorType StringToOperatorType(const std::string &operatorName)
    {
        OPERATOR_TYPE_FROM_STRING(Add)
        OPERATOR_TYPE_FROM_STRING(Conv)
        OPERATOR_TYPE_FROM_STRING(Constant)
        OPERATOR_TYPE_FROM_STRING(Sub)
        OPERATOR_TYPE_FROM_STRING(Reshape)
        OPERATOR_TYPE_FROM_STRING(Split)
        OPERATOR_TYPE_FROM_STRING(Concat)
        OPERATOR_TYPE_FROM_STRING(MatMul)
        OPERATOR_TYPE_FROM_STRING(Div)
        OPERATOR_TYPE_FROM_STRING(Mul)
        OPERATOR_TYPE_FROM_STRING(Sigmoid)
        OPERATOR_TYPE_FROM_STRING(Slice)
        OPERATOR_TYPE_FROM_STRING(Gather)
        OPERATOR_TYPE_FROM_STRING(Shape)
        OPERATOR_TYPE_FROM_STRING(Softmax)
        OPERATOR_TYPE_FROM_STRING(Transpose)
        OPERATOR_TYPE_FROM_STRING(Resize)
        OPERATOR_TYPE_FROM_STRING(MaxPool)

        if (operatorName == "Unknown")
        {
            return OperatorType::Unknown;
        }

        throw std::runtime_error("Unsupported operator type: " + operatorName);
    }

#undef OPERATOR_TYPE_TO_STRING

}
