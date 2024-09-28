#ifdef USE_HIP
#include "operator/operators.hpp"
#include "operator/elementwise_operator.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#include "utils.hpp"

namespace HIP_OP
{
    template <typename T>
    __global__ void concat_kernel(const T *input_data, T *output_data,
                                  size_t outer_size, size_t axis_size_input, size_t axis_size_output, size_t inner_size,
                                  size_t axis_offset, size_t num_input_elements)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_input_elements)
            return;

        // Compute indices
        size_t axis_size_input_inner = axis_size_input * inner_size;
        size_t outer_index = idx / axis_size_input_inner;
        size_t tmp = idx % axis_size_input_inner;
        size_t axis_index = tmp / inner_size;
        size_t inner_index = tmp % inner_size;

        // Compute input index
        size_t input_idx = idx;

        // Compute output index
        size_t axis_size_output_inner = axis_size_output * inner_size;
        size_t output_idx = outer_index * axis_size_output_inner + (axis_offset + axis_index) * inner_size + inner_index;

        output_data[output_idx] = input_data[input_idx];
    }

    template <typename T>
    OperatorExecuteResult executeConcatHIP(const std::vector<Tensor> &inputs, Tensor *output, int64_t axis)
    {
        size_t rank = output->getNDim();
        if (axis < 0)
        {
            axis += static_cast<int64_t>(rank);
        }

        // Compute outer_size (product of dimensions before axis)
        size_t outer_size = 1;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i)
        {
            outer_size *= output->getDims()[i];
        }

        // Compute inner_size (product of dimensions after axis)
        size_t inner_size = 1;
        for (size_t i = static_cast<size_t>(axis) + 1; i < rank; ++i)
        {
            inner_size *= output->getDims()[i];
        }

        // The total size along axis in output tensor
        size_t axis_size_output = output->getDims()[axis];

        T *output_data = static_cast<T *>(output->getDataPointer());
        if (!output_data)
        {
            return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
        }

        size_t axis_offset = 0;

        for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx)
        {
            const Tensor &input = inputs[input_idx];
            const T *input_data = static_cast<const T *>(input.getDataPointer());
            if (!input_data)
            {
                return OperatorExecuteResult::INPUT_TENSOR_ERROR;
            }

            // Get the size along the concatenation axis for this input tensor
            size_t axis_size_input = input.getDims()[axis];

            // Compute number of elements in input tensor
            size_t num_input_elements = input.getNumElements();

            int threadsPerBlock = 256;
            int blocksPerGrid = (num_input_elements + threadsPerBlock - 1) / threadsPerBlock;

            hipLaunchKernelGGL(concat_kernel<T>, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                               input_data, output_data,
                               outer_size, axis_size_input, axis_size_output, inner_size,
                               axis_offset, num_input_elements);

            /// FIXME: Implement it in the macro
            if (hipGetLastError() != hipSuccess)
            {
                return OperatorExecuteResult::HIP_ERROR;
            }

            axis_offset += axis_size_input;
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult ConcatOperatorImpl::execute(
        const std::vector<Tensor> &inputs,
        std::vector<Tensor *> &outputs,
        const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {

        int64_t axis = std::get<int64_t>(attributes.at("axis"));
        size_t rank = inputs[0].getNDim();

        // Adjust negative axis
        if (axis < 0)
        {
            axis += static_cast<int64_t>(rank);
        }

        // Ensure that the output tensor is allocated
        Tensor *output = outputs[0];
        size_t output_num_elements = output->getNumElements();
        if (!output->getBuffer() || output->getNumElements() != output_num_elements)
        {
            output->allocateBuffer(output->getDataType(), output_num_elements);
        }

        TensorDataType data_type = inputs[0].getDataType();

        switch (data_type)
        {
        case TensorDataType::FLOAT32:
            return executeConcatHIP<float>(inputs, output, axis);
        case TensorDataType::FLOAT64:
            return executeConcatHIP<double>(inputs, output, axis);
        case TensorDataType::INT32:
            return executeConcatHIP<int32_t>(inputs, output, axis);
        case TensorDataType::INT64:
            return executeConcatHIP<int64_t>(inputs, output, axis);
        case TensorDataType::INT8:
            return executeConcatHIP<int8_t>(inputs, output, axis);
        case TensorDataType::UINT8:
            return executeConcatHIP<uint8_t>(inputs, output, axis);
        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }
};
#endif
