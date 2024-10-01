#ifdef USE_HIP

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

#include "operator/operators.hpp"
#include "utils.hpp"

#define BLOCK_SIZE 256

namespace HIP_OP
{
    template <typename T>
    __device__ T exp_function(T x)
    {
        return exp(x);
    }

    template <>
    __device__ __half exp_function<__half>(__half x)
    {
        return __float2half(expf(__half2float(x)));
    }

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

        T max_val = input_data[offset];
        for (size_t i = 1; i < axis_size; ++i)
        {
            size_t idx = offset + i * inner_size;
            if (input_data[idx] > max_val)
            {
                max_val = input_data[idx];
            }
        }

        T sum = static_cast<T>(0.0f);
        for (size_t i = 0; i < axis_size; ++i)
        {
            size_t idx = offset + i * inner_size;
            output_data[idx] = exp_function(input_data[idx] - max_val);
            sum = sum + output_data[idx];
        }

        if (static_cast<float>(sum) > 0.0f)
        {
            for (size_t i = 0; i < axis_size; ++i)
            {
                size_t idx = offset + i * inner_size;
                output_data[idx] = output_data[idx] / sum;
            }
        }
    }

    template <typename T>
    OperatorExecuteResult executeSoftmax(const Tensor &input, Tensor *output,
                                         size_t outer_size, size_t axis_size, size_t inner_size)
    {
        const T *input_data = input.data<T>();
        T *output_data = output->data<T>();

        size_t num_blocks = (inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dim3 blockDim(BLOCK_SIZE);
        dim3 gridDim(outer_size * num_blocks);

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

        const void *input_data = input.getDataPointer();
        void *output_data = output->getDataPointer();

        size_t num_blocks = (inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dim3 blockDim(BLOCK_SIZE);
        dim3 gridDim(outer_size * num_blocks);

        switch (dataType)
        {
        case TensorDataType::FLOAT32:
            return executeSoftmax<float>(input, output, outer_size, axis_size, inner_size);
        case TensorDataType::FLOAT64:
            return executeSoftmax<double>(input, output, outer_size, axis_size, inner_size);
        case TensorDataType::FLOAT16:
            return executeSoftmax<half_t>(input, output, outer_size, axis_size, inner_size);

        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
};

#endif
