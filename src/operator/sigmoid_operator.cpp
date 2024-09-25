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

template <typename T>
OperatorExecuteResult executeSigmoidTyped(const std::vector<Tensor> &inputs, Tensor *output,
                                          const std::vector<size_t> &input_strides,
                                          const std::vector<size_t> &output_strides,
                                          const std::vector<size_t> &output_shape)
{
    size_t num_elements = 1;
    for (size_t dim : output_shape)
    {
        num_elements *= dim;
    }

    T *output_data = new (std::nothrow) T[num_elements];
    if (!output_data)
    {
        return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
    }

    const T *input_data = inputs[0].data<T>();
    size_t num_dims = output_shape.size();

    for (size_t idx = 0; idx < num_elements; ++idx)
    {
        size_t remainder = idx;
        std::vector<size_t> indices(num_dims, 0);
        for (size_t dim = 0; dim < num_dims; ++dim)
        {
            indices[dim] = remainder / output_strides[dim];
            remainder = remainder % output_strides[dim];
        }

        size_t offset = 0;
        for (size_t dim = 0; dim < num_dims; ++dim)
        {
            size_t input_dim_size = inputs[0].getDims()[dim];
            size_t index = (input_dim_size == 1) ? 0 : indices[dim];
            offset += index * input_strides[dim];
        }

        T value = 1 / (1 + std::exp(-input_data[offset]));
        output_data[idx] = value;
    }

    output->setDataPointer<T>(output_data, output_shape);
    return OperatorExecuteResult::SUCCESS;
}

OperatorExecuteResult SigmoidOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                               const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() != 1)
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    const TensorDataType dataType = inputs[0].getDataType();

    std::vector<size_t> output_shape = inputs[0].getDims();
    if (output_shape.empty())
    {
        return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
    }

    if (outputs.empty() || outputs[0] == nullptr)
    {
        return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
    }

    Tensor *output = outputs[0];

    size_t num_dims = output_shape.size();
    std::vector<size_t> input_strides(num_dims, 1);
    std::vector<size_t> output_strides(num_dims, 1);

    for (int i = num_dims - 2; i >= 0; --i)
    {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    const auto &input_shape = inputs[0].getDims();
    for (int i = num_dims - 2; i >= 0; --i)
    {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    switch (dataType)
    {
    case TensorDataType::FLOAT32:
        return executeSigmoidTyped<float>(inputs, output, input_strides, output_strides, output_shape);
    case TensorDataType::FLOAT64:
        return executeSigmoidTyped<double>(inputs, output, input_strides, output_strides, output_shape);

    default:
        return OperatorExecuteResult::UNSUPPORTED_OPERATION;
    }
}
