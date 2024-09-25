#include "operator/operators.hpp"
#include <iostream>
#include <type_traits>
namespace CPU_OP
{

    template <typename T>
    OperatorExecuteResult executeTyped(const Tensor &data, const Tensor &indices, Tensor *output, int axis)
    {
        // Get the shapes of input tensors
        const std::vector<size_t> &data_shape = data.getDims();
        const std::vector<size_t> &indices_shape = indices.getDims();
        size_t data_rank = data_shape.size();
        size_t indices_rank = indices_shape.size();

        // Determine the shape of the output tensor
        std::vector<size_t> output_shape;
        output_shape.reserve(indices_rank + data_rank - 1);

        // Copy the shape of indices into output shape
        output_shape.insert(output_shape.end(), indices_shape.begin(), indices_shape.end());

        // Copy the remaining shape from data excluding the specified axis
        for (size_t i = 0; i < data_rank; ++i)
        {
            if (i != static_cast<size_t>(axis))
            {
                output_shape.push_back(data_shape[i]);
            }
        }

        // Allocate the output tensor memory
        size_t num_output_elements = 1;
        for (size_t dim : output_shape)
        {
            num_output_elements *= dim;
        }

        T *output_data = new (std::nothrow) T[num_output_elements];
        if (!output_data)
        {
            return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
        }

        // Perform the gather operation based on the axis
        const T *data_ptr = data.data<T>();
        const int64_t *indices_ptr = indices.data<int64_t>();

        // Calculate strides for the data tensor
        std::vector<size_t> data_strides(data_rank, 1);
        for (int i = data_rank - 2; i >= 0; --i)
        {
            data_strides[i] = data_strides[i + 1] * data_shape[i + 1];
        }

        // Iterate over each element in the indices tensor and gather data
        size_t output_offset = 0;
        for (size_t i = 0; i < indices.getNumElements(); ++i)
        {
            int64_t index = indices_ptr[i];
            if (index < -static_cast<int64_t>(data_shape[axis]) || index >= static_cast<int64_t>(data_shape[axis]))
            {
                delete[] output_data;
                return OperatorExecuteResult::INPUT_TENSOR_ERROR;
            }

            // Adjust negative indices
            if (index < 0)
            {
                index += data_shape[axis];
            }

            // Compute the multi-dimensional offset in the data tensor
            size_t data_offset = index * data_strides[axis];

            // Compute the strides for copying elements from the data tensor
            size_t inner_dim_size = 1;
            for (size_t dim = axis + 1; dim < data_rank; ++dim)
            {
                inner_dim_size *= data_shape[dim];
            }

            // Copy elements from data tensor to output tensor
            for (size_t j = 0; j < inner_dim_size; ++j)
            {
                output_data[output_offset++] = data_ptr[data_offset + j];
            }
        }

        // Set data pointer and shape for output tensor
        output->setDataType(data.getDataType());
        output->setDataPointer<T>(output_data, output_shape);

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult GatherOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                      const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        if (inputs.size() != 2)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        const Tensor &data = inputs[0];
        const Tensor &indices = inputs[1];

        int64_t axis = 0;
        if (attributes.find("axis") != attributes.end())
        {
            axis = std::get<int64_t>(attributes.at("axis"));
        }

        size_t data_rank = data.getDims().size();
        if (axis < 0)
        {
            axis += data_rank;
        }

        if (axis < 0 || static_cast<size_t>(axis) >= data_rank)
        {
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        }

        // Check that output tensor is available
        if (outputs.empty() || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        Tensor *output = outputs[0];

        switch (data.getDataType())
        {
        case TensorDataType::FLOAT32:
            return executeTyped<float>(data, indices, output, axis);
        case TensorDataType::INT32:
            return executeTyped<int32_t>(data, indices, output, axis);
        case TensorDataType::INT64:
            return executeTyped<int64_t>(data, indices, output, axis);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }

}