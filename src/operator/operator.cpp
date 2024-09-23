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

/// FIXME: Don't use switch-case for operator creation
///        Use a map of operator type to operator class
std::unique_ptr<Operator> OperatorFactory::createOperator(const std::string &op_type)
{
    if (op_type == "Add")
    {
        return std::make_unique<AddOperator>();
    }
    else if (op_type == "Sub")
    {
        return std::make_unique<SubOperator>();
    }
    else if (op_type == "Mul")
    {
        return std::make_unique<MulOperator>();
    }
    else if (op_type == "Div")
    {
        return std::make_unique<DivOperator>();
    }
    else if (op_type == "MatMul")
    {
        return std::make_unique<MatMulOperator>();
    }
    else if (op_type == "Conv")
    {
        return std::make_unique<ConvOperator>();
    }
    else if (op_type == "Sigmoid")
    {
        return std::make_unique<SigmoidOperator>();
    }
    else if (op_type == "Reshape")
    {
        return std::make_unique<ReshapeOperator>();
    }
    else if (op_type == "Constant")
    {
        return std::make_unique<ConstantOperator>();
    }
    else if (op_type == "Split")
    {
        return std::make_unique<SplitOperator>();
    }
    else if (op_type == "Concat")
    {
        return std::make_unique<ConcatOperator>();
    }
    else if (op_type == "Gather")
    {
        return std::make_unique<GatherOperator>();
    }
    else if (op_type == "Shape")
    {
        return std::make_unique<ShapeOperator>();
    }
    else if (op_type == "Softmax")
    {
        return std::make_unique<SoftmaxOperator>();
    }
    else if (op_type == "Transpose")
    {
        return std::make_unique<TransposeOperator>();
    }
    else if (op_type == "Slice")
    {
        return std::make_unique<SliceOperator>();
    }
    else if (op_type == "Resize")
    {
        return std::make_unique<ResizeOperator>();
    }
    else if (op_type == "MaxPool")
    {
        return std::make_unique<MaxPoolOperator>();
    }
    else
    {
        throw std::runtime_error("Unsupported operator type: " + op_type);
    }
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
        case OperatorExecuteResult::UNKNOWN_ERROR:
            return "UNKNOWN_ERROR";

        default:
            return "UNKNOWN";
        }
    }
}