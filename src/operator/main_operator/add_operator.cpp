#include <iostream>
#include <type_traits>

#include "operator/operators.hpp"
#include "operator/aux_operator/elementwise_operator.hpp"

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

template <typename T>
OperatorExecuteResult executeAddTyped(const std::vector<Tensor> &inputs, Tensor *output,
                                      const std::vector<std::vector<size_t>> &input_strides,
                                      const std::vector<size_t> &output_strides,
                                      const std::vector<size_t> &output_shape)
{
    const size_t num_elements = output->getNumElements();
    T *output_data = new (std::nothrow) T[num_elements];
    if (!output_data)
    {
        return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
    }

    std::vector<const T *> input_data_ptrs(inputs.size());
    for (size_t idx = 0; idx < inputs.size(); ++idx)
    {
        input_data_ptrs[idx] = inputs[idx].data<T>();
    }

    const size_t num_dims = output_shape.size();
    std::vector<size_t> indices(num_dims, 0);

    for (size_t idx = 0; idx < num_elements; ++idx)
    {
        size_t remainder = idx;
        for (size_t dim = 0; dim < num_dims; ++dim)
        {
            indices[dim] = remainder / output_strides[dim];
            remainder = remainder % output_strides[dim];
        }

        T sum = static_cast<T>(0);
        for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx)
        {
            const auto &input_shape = inputs[input_idx].getDims();
            size_t input_rank = input_shape.size();

            std::vector<size_t> adjusted_shape = input_shape;
            if (input_rank < num_dims)
            {
                adjusted_shape.insert(adjusted_shape.begin(), num_dims - input_rank, 1);
            }

            size_t offset = 0;
            for (size_t dim = 0; dim < num_dims; ++dim)
            {
                size_t input_dim_size = adjusted_shape[dim];
                size_t index = (input_dim_size == 1) ? 0 : indices[dim];
                offset += index * input_strides[input_idx][dim];
            }

            sum += input_data_ptrs[input_idx][offset];
        }
        output_data[idx] = sum;
    }

    output->setDataType(inputs[0].getDataType());
    output->setDataPointer<T>(output_data, output_shape);

    return OperatorExecuteResult::SUCCESS;
}

OperatorExecuteResult AddOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() < 2)
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    const TensorDataType dataType = inputs[0].getDataType();
    for (size_t i = 1; i < inputs.size(); i++)
    {
        if (inputs[i].getDataType() != dataType)
        {
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }

    std::vector<std::vector<size_t>> input_shapes;
    for (const auto &tensor : inputs)
    {
        input_shapes.push_back(tensor.getDims());
    }
    std::vector<size_t> output_shape = compute_broadcast_shape(input_shapes);
    if (output_shape.empty())
    {
        return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
    }

    if (outputs.empty() || outputs[0] == nullptr)
    {
        return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
    }

    Tensor *output = outputs[0];
    output->reshape(output_shape);
    size_t num_dims = output->getNDim();
    std::vector<size_t> output_strides(num_dims, 1);
    for (int i = num_dims - 2; i >= 0; --i)
    {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    std::vector<std::vector<size_t>> input_strides(inputs.size());
    for (size_t idx = 0; idx < inputs.size(); ++idx)
    {
        input_strides[idx] = compute_broadcast_strides(inputs[idx].getDims(), output_shape);
    }

    switch (dataType)
    {
    case TensorDataType::FLOAT32:
        return executeElementwiseOperation<float>(inputs, output, input_strides, output_strides, output_shape,
                                                  std::plus<float>());
    case TensorDataType::FLOAT64:
        return executeElementwiseOperation<double>(inputs, output, input_strides, output_strides, output_shape,
                                                   std::plus<double>());
    case TensorDataType::INT32:
        return executeElementwiseOperation<int32_t>(inputs, output, input_strides, output_strides, output_shape,
                                                    std::plus<int32_t>());
    case TensorDataType::INT64:
        return executeElementwiseOperation<int64_t>(inputs, output, input_strides, output_strides, output_shape,
                                                    std::plus<int64_t>());
    case TensorDataType::INT8:
        return executeElementwiseOperation<int8_t>(inputs, output, input_strides, output_strides, output_shape,
                                                   std::plus<int8_t>());
    case TensorDataType::UINT8:
        return executeElementwiseOperation<uint8_t>(inputs, output, input_strides, output_strides, output_shape,
                                                    std::plus<uint8_t>());
    default:
        return OperatorExecuteResult::UNSUPPORTED_OPERATION;
    }
}
