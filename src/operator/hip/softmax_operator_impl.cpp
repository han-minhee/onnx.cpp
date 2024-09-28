#ifdef USE_HIP

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <math.h>

#include "operator/operators.hpp"
#include "utils.hpp"

namespace HIP_OP
{
    template <typename T>
    __global__ void softmax_kernel(const T *input_data, T *output_data,
                                   size_t outer_size, size_t axis_size, size_t inner_size)
    {
        size_t total_blocks = gridDim.x / outer_size;
        size_t outer = blockIdx.x % outer_size;
        size_t block = blockIdx.x / outer_size;
        size_t thread = threadIdx.x;

        size_t inner = block * blockDim.x + thread;

        if (outer >= outer_size || inner >= inner_size)
            return;

        size_t offset = outer * axis_size * inner_size + inner;

        // Step 1: Find the max value for numerical stability across the entire axis
        T max_val = input_data[offset];
        for (size_t i = 1; i < axis_size; ++i)
        {
            size_t idx = offset + i * inner_size;
            if (input_data[idx] > max_val)
            {
                max_val = input_data[idx];
            }
        }

        // Step 2: Calculate the exponential values and sum them
        T sum = 0.0f;
        for (size_t i = 0; i < axis_size; ++i)
        {
            size_t idx = offset + i * inner_size;
            output_data[idx] = exp(input_data[idx] - max_val);
            sum += output_data[idx];
        }

        // Step 3: Normalize the values
        if (sum > 0) // Avoid potential division by zero
        {
            for (size_t i = 0; i < axis_size; ++i)
            {
                size_t idx = offset + i * inner_size;
                output_data[idx] /= sum;
            }
        }
    }

    template <typename T>
    OperatorExecuteResult executeSoftmax(const Tensor &input, Tensor *output,
                                         size_t outer_size, size_t axis_size, size_t inner_size)
    {
        const T *input_data = input.data<T>();
        T *output_data = output->data<T>();

        const int BLOCK_SIZE = 256;

        // Calculate the number of blocks needed to cover all inner indices
        size_t num_blocks = (inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dim3 blockDim(BLOCK_SIZE);
        dim3 gridDim(outer_size * num_blocks); // Each outer slice can have multiple blocks

        // Launch the kernel with updated grid dimensions
        hipLaunchKernelGGL(softmax_kernel<T>, gridDim, blockDim, 0, 0,
                           input_data, output_data, outer_size, axis_size, inner_size);

        hipErrorCheck(hipDeviceSynchronize());

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult SoftmaxOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {

        const Tensor &input = inputs[0];
        Tensor *output = outputs[0];

        TensorDataType dataType = input.getDataType();

        int64_t axis = -1;
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
        if (axis < 0 || static_cast<size_t>(axis) >= rank)
        {
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        }

        size_t outer_size = 1;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i)
        {
            outer_size *= input_shape[i];
        }

        size_t axis_size = input_shape[axis];

        size_t inner_size = 1;
        for (size_t i = static_cast<size_t>(axis) + 1; i < rank; ++i)
        {
            inner_size *= input_shape[i];
        }

        // Execute softmax based on data type
        switch (dataType)
        {
        case TensorDataType::FLOAT32:
            return executeSoftmax<float>(input, output, outer_size, axis_size, inner_size);

        // Add cases for other data types as needed
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
};

#endif
