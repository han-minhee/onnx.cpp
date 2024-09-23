#include "operator/operators.hpp"
#include <iostream>
#include <algorithm>

// Modify the inferOutputShapes function
std::vector<std::vector<size_t>> ConcatOperator::inferOutputShapes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {};
    }

    // Get 'axis' attribute
    if (attributes.find("axis") == attributes.end())
    {
        return {}; // axis is required
    }

    int64_t axis = std::get<int64_t>(attributes.at("axis"));
    size_t rank = inputs[0].getNDim();

    // Adjust negative axis
    if (axis < 0)
    {
        axis += static_cast<int64_t>(rank);
    }

    if (axis < 0 || axis >= static_cast<int64_t>(rank))
    {
        return {}; // Invalid axis
    }

    // Initialize output shape with the shape of the first input
    std::vector<size_t> output_shape = inputs[0].getDims();

    // Sum the sizes along the concatenation axis
    size_t axis_size = output_shape[axis];

    for (size_t i = 1; i < inputs.size(); ++i)
    {
        const auto &input_shape = inputs[i].getDims();

        // Check that all inputs have the same rank
        if (input_shape.size() != rank)
        {
            return {}; // Shape mismatch
        }

        // Check that dimensions match except along the concatenation axis
        for (size_t dim = 0; dim < rank; ++dim)
        {
            if (dim == static_cast<size_t>(axis))
            {
                continue; // Skip concatenation axis
            }
            if (input_shape[dim] != output_shape[dim])
            {
                return {}; // Return empty to indicate a shape mismatch
            }
        }

        // Accumulate size along the axis
        axis_size += input_shape[axis];
    }

    // Set the concatenated size along the axis
    output_shape[axis] = axis_size;

    return {output_shape};
}

std::vector<TensorDataType> ConcatOperator::inferOutputDataTypes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {TensorDataType::UNDEFINED};
    }

    // Check that all inputs have the same data type
    TensorDataType data_type = inputs[0].getDataType();
    for (const auto &input : inputs)
    {
        if (input.getDataType() != data_type)
        {
            return {TensorDataType::UNDEFINED};
        }
    }

    return {data_type};
}

template <typename T>
OperatorExecuteResult executeConcatTyped(const std::vector<Tensor> &inputs, Tensor *output, int64_t axis)
{
    size_t rank = output->getNDim();
    size_t adjusted_axis = static_cast<size_t>(axis);

    // Compute outer_size (product of dimensions before axis)
    size_t outer_size = 1;
    for (size_t i = 0; i < adjusted_axis; ++i)
    {
        outer_size *= output->getDims()[i];
    }

    // Compute inner_size (product of dimensions after axis)
    size_t inner_size = 1;
    for (size_t i = adjusted_axis + 1; i < rank; ++i)
    {
        inner_size *= output->getDims()[i];
    }

    // Allocate memory for the output tensor
    size_t output_num_elements = output->getNumElements();
    T *output_data = new (std::nothrow) T[output_num_elements];
    if (!output_data)
    {
        return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
    }

    // Perform concatenation
    size_t output_offset = 0;
    for (size_t outer_index = 0; outer_index < outer_size; ++outer_index)
    {
        for (const auto &input : inputs)
        {
            const T *input_data = input.data<T>();
            if (!input_data)
            {
                delete[] output_data;
                return OperatorExecuteResult::INPUT_TENSOR_ERROR;
            }

            size_t input_axis_dim_size = input.getDims()[adjusted_axis];
            size_t copy_size = input_axis_dim_size * inner_size;

            size_t input_offset = outer_index * input_axis_dim_size * inner_size;

            std::copy(input_data + input_offset, input_data + input_offset + copy_size, output_data + output_offset);

            output_offset += copy_size;
        }
    }

    output->setDataType(inputs[0].getDataType());

    // Create a copy of the dimensions before calling setDataPointer
    std::vector<size_t> output_dims = output->getDims();
    output->setDataPointer<T>(output_data, output_dims);

    return OperatorExecuteResult::SUCCESS;
}

OperatorExecuteResult ConcatOperator::execute(
    const std::vector<Tensor> &inputs,
    std::vector<Tensor *> &outputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    // Get 'axis' attribute
    if (attributes.find("axis") == attributes.end())
    {
        return OperatorExecuteResult::ATTRIBUTE_ERROR; // axis is required
    }

    int64_t axis = std::get<int64_t>(attributes.at("axis"));
    size_t rank = inputs[0].getNDim();

    // Adjust negative axis
    if (axis < 0)
    {
        axis += static_cast<int64_t>(rank);
    }

    if (axis < 0 || axis >= static_cast<int64_t>(rank))
    {
        return OperatorExecuteResult::ATTRIBUTE_ERROR; // Invalid axis
    }

    // Check that all inputs have the same data type and compatible shapes
    TensorDataType data_type = inputs[0].getDataType();
    for (const auto &input : inputs)
    {
        if (input.getDataType() != data_type)
        {
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        if (input.getNDim() != rank)
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        const auto &input_shape = input.getDims();
        const auto &reference_shape = inputs[0].getDims();

        for (size_t dim = 0; dim < rank; ++dim)
        {
            if (dim == static_cast<size_t>(axis))
            {
                continue; // Skip concatenation axis
            }
            if (input_shape[dim] != reference_shape[dim])
            {
                return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
            }
        }
    }

    if (outputs.empty() || outputs[0] == nullptr)
    {
        return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
    }

    Tensor *output = outputs[0];

    // Compute output shape
    auto output_shapes = inferOutputShapes(inputs, attributes);
    if (output_shapes.empty())
    {
        return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
    }
    output->reshape(output_shapes[0]);

    // Call the typed execute function
    switch (data_type)
    {
    case TensorDataType::FLOAT32:
        return executeConcatTyped<float>(inputs, output, axis);
    case TensorDataType::FLOAT64:
        return executeConcatTyped<double>(inputs, output, axis);
    case TensorDataType::INT32:
        return executeConcatTyped<int32_t>(inputs, output, axis);
    case TensorDataType::INT64:
        return executeConcatTyped<int64_t>(inputs, output, axis);
    case TensorDataType::INT8:
        return executeConcatTyped<int8_t>(inputs, output, axis);
    case TensorDataType::UINT8:
        return executeConcatTyped<uint8_t>(inputs, output, axis);
    default:
        return OperatorExecuteResult::UNSUPPORTED_OPERATION;
    }
}
