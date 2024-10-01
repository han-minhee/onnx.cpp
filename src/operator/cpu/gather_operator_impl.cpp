#include "operator/operators.hpp"
#include <iostream>
#include <type_traits>
namespace CPU_OP
{

    template <typename T>
    OperatorExecuteResult executeGather(const Tensor &input, const Tensor &indices, Tensor *output, int axis)
    {
        const std::vector<size_t> &input_shape = input.getDims();
        size_t input_rank = input_shape.size();

        T *output_data = output->data<T>();
        if (!output_data)
        {
            return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
        }

        const T *input_values = input.data<T>();
        const int64_t *indices_values = indices.data<int64_t>();

        std::vector<size_t> data_strides = input.getStrides();

        size_t output_offset = 0;
        for (size_t i = 0; i < indices.getNumElements(); ++i)
        {
            int64_t index = indices_values[i];
            if (index < -static_cast<int64_t>(input_shape[axis]) || index >= static_cast<int64_t>(input_shape[axis]))
            {
                return OperatorExecuteResult::INPUT_TENSOR_ERROR;
            }

            if (index < 0)
            {
                index += input_shape[axis];
            }

            size_t data_offset = index * data_strides[axis];

            size_t inner_dim_size = 1;
            for (size_t dim = axis + 1; dim < input_rank; ++dim)
            {
                inner_dim_size *= input_shape[dim];
            }

            for (size_t j = 0; j < inner_dim_size; ++j)
            {
                output_data[output_offset++] = input_values[data_offset + j];
            }
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult GatherOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                      const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        const Tensor &input = inputs[0];
        const Tensor &indices = inputs[1];

        int64_t axis = 0;
        if (attributes.find("axis") != attributes.end())
        {
            axis = std::get<int64_t>(attributes.at("axis"));
        }

        size_t input_rank = input.getDims().size();
        if (axis < 0)
        {
            axis += input_rank;
        }

        Tensor *output = outputs[0];

        switch (input.getDataType())
        {
        case TensorDataType::FLOAT32:
            return executeGather<float>(input, indices, output, axis);
        case TensorDataType::INT32:
            return executeGather<int32_t>(input, indices, output, axis);
        case TensorDataType::INT64:
            return executeGather<int64_t>(input, indices, output, axis);
        case TensorDataType::INT8:
            return executeGather<int8_t>(input, indices, output, axis);
        case TensorDataType::UINT8:
            return executeGather<uint8_t>(input, indices, output, axis);
        case TensorDataType::FLOAT16:
            return executeGather<half_t>(input, indices, output, axis);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
}