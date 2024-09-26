#include "operator/operators.hpp"
#include <iostream>
#include <stdexcept>

namespace CPU_OP
{

    template <typename T>
    OperatorExecuteResult executeSplit(const Tensor &input, std::vector<Tensor *> &outputs,
                                       const std::vector<std::vector<size_t>> &output_shapes,
                                       const std::vector<int64_t> &split_sizes, int64_t axis)
    {
        const T *input_data = input.data<T>();

        if (!input_data)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        const std::vector<size_t> &input_shape = input.getDims();
        size_t rank = input_shape.size();

        size_t outer_dim = 1;
        for (size_t dim = 0; dim < static_cast<size_t>(axis); ++dim)
        {
            outer_dim *= input_shape[dim];
        }

        size_t inner_dim = 1;
        for (size_t dim = static_cast<size_t>(axis) + 1; dim < rank; ++dim)
        {
            inner_dim *= input_shape[dim];
        }

        size_t input_axis_dim = input_shape[axis];
        size_t offset = 0;
        size_t num_splits = split_sizes.size();

        for (size_t i = 0; i < num_splits; ++i)
        {
            size_t split_size = static_cast<size_t>(split_sizes[i]);
            Tensor *output = outputs[i];

            if (output == nullptr)
            {
                return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
            }

            const std::vector<size_t> &output_shape = output_shapes[i];
            size_t num_elements = outer_dim * split_size * inner_dim;

            // Allocate the output buffer if not allocated or if dimensions mismatch
            if (!output->data<T>() || output->getNumElements() != num_elements)
            {
                output->allocateBuffer(input.getDataType(), num_elements);
                output->reshape(output_shape);
            }

            T *output_data = output->data<T>();
            if (!output_data)
            {
                return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
            }

            // Copy data from input to output
            for (size_t outer = 0; outer < outer_dim; ++outer)
            {
                for (size_t k = 0; k < split_size; ++k)
                {
                    size_t input_axis_index = offset + k;
                    for (size_t inner = 0; inner < inner_dim; ++inner)
                    {
                        size_t input_idx = outer * input_axis_dim * inner_dim + input_axis_index * inner_dim + inner;
                        size_t output_idx = outer * split_size * inner_dim + k * inner_dim + inner;

                        output_data[output_idx] = input_data[input_idx];
                    }
                }
            }

            offset += split_size;
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult SplitOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        if (inputs.empty())
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        std::vector<std::vector<size_t>> output_shapes;
        for (const Tensor *output : outputs)
        {
            if (output == nullptr)
            {
                return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
            }
            output_shapes.push_back(output->getDims());
        }

        const Tensor &input = inputs.at(0);

        int64_t axis = 0;
        if (attributes.count("axis"))
        {
            axis = std::get<int64_t>(attributes.at("axis"));
        }

        const std::vector<size_t> &input_shape = input.getDims();
        size_t rank = input_shape.size();

        if (axis < 0)
        {
            axis += rank;
        }

        std::vector<int64_t> split_sizes(output_shapes.size());
        for (size_t i = 0; i < output_shapes.size(); ++i)
        {
            split_sizes[i] = static_cast<int64_t>(output_shapes[i][axis]);
        }

        // Use appropriate data type for execution
        switch (input.getDataType())
        {
        case TensorDataType::FLOAT32:
            return executeSplit<float>(input, outputs, output_shapes, split_sizes, axis);
        case TensorDataType::FLOAT64:
            return executeSplit<double>(input, outputs, output_shapes, split_sizes, axis);
        case TensorDataType::INT32:
            return executeSplit<int32_t>(input, outputs, output_shapes, split_sizes, axis);
        case TensorDataType::INT64:
            return executeSplit<int64_t>(input, outputs, output_shapes, split_sizes, axis);
        case TensorDataType::INT8:
            return executeSplit<int8_t>(input, outputs, output_shapes, split_sizes, axis);
        case TensorDataType::UINT8:
            return executeSplit<uint8_t>(input, outputs, output_shapes, split_sizes, axis);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
}
