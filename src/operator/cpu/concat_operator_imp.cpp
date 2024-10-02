#include "operator/operators.hpp"

namespace CPU_OP
{
    template <typename T>
    OperatorExecuteResult executeConcat(const std::vector<Tensor> &inputs, Tensor *output, int64_t axis)
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

        size_t output_num_elements = output->getNumElements();

        T *output_data = output->data<T>(); // Access the buffer's data
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
                    return OperatorExecuteResult::INPUT_TENSOR_ERROR;
                }
                size_t input_axis_dim_size = input.getDims()[adjusted_axis];
                size_t copy_size = input_axis_dim_size * inner_size;
                size_t input_offset = outer_index * input_axis_dim_size * inner_size;
                std::copy(input_data + input_offset, input_data + input_offset + copy_size, output_data + output_offset);
                output_offset += copy_size;
            }
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult ConcatOperatorImpl::execute(
        const std::vector<Tensor> &inputs,
        std::vector<Tensor *> &outputs,
        const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        int64_t axis = std::get<int64_t>(attributes.at("axis"));
        size_t rank = inputs[0].getNDim();

        // Adjust negative axis
        if (axis < 0)
        {
            axis += static_cast<int64_t>(rank);
        }

        // XXX: output tensor should already have a shape and data type defined
        Tensor *output = outputs[0];
        TensorDataType data_type = inputs[0].getDataType();

        switch (data_type)
        {
        case TensorDataType::FLOAT32:
            return executeConcat<float>(inputs, output, axis);
        case TensorDataType::FLOAT64:
            return executeConcat<double>(inputs, output, axis);
        case TensorDataType::INT32:
            return executeConcat<int32_t>(inputs, output, axis);
        case TensorDataType::INT64:
            return executeConcat<int64_t>(inputs, output, axis);
        case TensorDataType::INT8:
            return executeConcat<int8_t>(inputs, output, axis);
        case TensorDataType::UINT8:
            return executeConcat<uint8_t>(inputs, output, axis);

        case TensorDataType::FLOAT16:
            return executeConcat<half_t>(inputs, output, axis);
        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }

}