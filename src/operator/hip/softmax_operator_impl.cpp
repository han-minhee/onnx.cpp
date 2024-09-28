#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <math.h>

#include "utils.hpp"

#define BLOCK_SIZE 256

namespace HIP_OP
{
    template <typename T>
    __global__ void softmax_kernel(const T *input, T *output, size_t inner_dim, size_t outer_dim)
    {
        size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (outer_idx >= outer_dim)
            return;

        // Find the start of the segment
        const T *input_segment = input + outer_idx * inner_dim;
        T *output_segment = output + outer_idx * inner_dim;

        // Compute max value for numerical stability
        T max_val = input_segment[0];
        for (size_t i = 1; i < inner_dim; ++i)
        {
            max_val = max(max_val, input_segment[i]);
        }

        // Compute exponentials and sum
        T sum_exp = 0;
        for (size_t i = 0; i < inner_dim; ++i)
        {
            output_segment[i] = exp(input_segment[i] - max_val);
            sum_exp += output_segment[i];
        }

        // Normalize to get softmax probabilities
        for (size_t i = 0; i < inner_dim; ++i)
        {
            output_segment[i] /= sum_exp;
        }
    }

    OperatorExecuteResult SoftmaxOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        if (inputs.size() != 1)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }
        if (outputs.empty() || outputs.size() != 1)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const Tensor &input = inputs[0];
        Tensor *output = outputs[0];

        TensorDataType dtype = input.getDataType();
        const auto &dims = input.getDims();
        
        size_t inner_dim = dims.back(); // The dimension along which to apply softmax
        size_t outer_dim = input.getNumElements() / inner_dim;

        const void *input_data = input.getDataPointer();
        void *output_data = output->getDataPointer();

        dim3 gridSize((outer_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(softmax_kernel<float>, gridSize, blockSize, 0, 0,
                                                    static_cast<const float *>(input_data),
                                                    static_cast<float *>(output_data),
                                                    inner_dim, outer_dim));
            break;
        case TensorDataType::FLOAT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(softmax_kernel<double>, gridSize, blockSize, 0, 0,
                                                    static_cast<const double *>(input_data),
                                                    static_cast<double *>(output_data),
                                                    inner_dim, outer_dim));
            break;
        // Add cases for other data types as needed
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        hipErrorCheck(hipDeviceSynchronize());

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif
