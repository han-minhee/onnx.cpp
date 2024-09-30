#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 256

namespace HIP_OP
{
    // The HIP kernel for gather operation
    template <typename T>
    __global__ void gather_kernel(const T *__restrict__ input_data, const size_t *__restrict__ input_shape,
                                  const int64_t *__restrict__ indices_data, T *__restrict__ output_data,
                                  size_t num_indices, size_t axis, size_t inner_dim, size_t outer_dim)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_indices * inner_dim * outer_dim)
            return;

        size_t output_outer_idx = idx / (num_indices * inner_dim);
        size_t index_idx = (idx / inner_dim) % num_indices;
        size_t output_inner_idx = idx % inner_dim;

        int64_t gather_index = indices_data[index_idx];
        if (gather_index < 0 || gather_index >= input_shape[axis])
            return;

        size_t input_idx = output_outer_idx * input_shape[axis] * inner_dim + gather_index * inner_dim + output_inner_idx;
        output_data[idx] = input_data[input_idx];
    }

    // The executeGather function that prepares data and launches the kernel
    template <typename T>
    OperatorExecuteResult executeGather(const Tensor &input, const Tensor &indices, Tensor *output, int axis)
    {
        const std::vector<size_t> &input_shape = input.getDims();
        size_t num_elements = output->getNumElements();
        size_t num_indices = indices.getNumElements();

        const void *input_data = input.getDataPointer();
        const void *indices_data = indices.getDataPointer();
        void *output_data = output->getDataPointer();

        // Calculating inner, outer, and axis dimensions
        size_t inner_dim = 1;
        size_t outer_dim = 1;
        for (size_t i = axis + 1; i < input_shape.size(); ++i)
        {
            inner_dim *= input_shape[i];
        }
        for (size_t i = 0; i < axis; ++i)
        {
            outer_dim *= input_shape[i];
        }

        // Prepare grid and block size
        dim3 gridSize((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        // Launch the kernel
        hipKernelLaunchCheck(hipLaunchKernelGGL(gather_kernel<T>, gridSize, blockSize, 0, 0,
                                                static_cast<const T *>(input_data), input.d_getDims(),
                                                static_cast<const int64_t *>(indices_data), static_cast<T *>(output_data),
                                                num_indices, axis, inner_dim, outer_dim));

        hipErrorCheck(hipDeviceSynchronize());

        return OperatorExecuteResult::SUCCESS;
    }

    // The execute function that handles different data types
    OperatorExecuteResult GatherOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                      const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
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
        TensorDataType data_type = input.getDataType();
        switch(data_type){
            case TensorDataType::FLOAT32:
                return executeGather<float>(input, indices, output, axis);
            case TensorDataType::FLOAT64:
                return executeGather<double>(input, indices, output, axis);
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
                return OperatorExecuteResult::NOT_IMPLEMENTED;
        }

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif
