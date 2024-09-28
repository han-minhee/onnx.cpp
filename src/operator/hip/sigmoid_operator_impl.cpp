#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#include "utils.hpp"

#define BLOCK_SIZE 256

namespace HIP_OP
{
    template <typename T>
    __global__ void sigmoid_kernel(const T *input, T *output, size_t num_elements)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements)
        {
            output[idx] = 1.0f / (1.0f + exp(-input[idx]));
        }
    }

    OperatorExecuteResult SigmoidOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {

        const Tensor &input = inputs[0];
        Tensor *output = outputs[0];

        TensorDataType dtype = input.getDataType();
        size_t num_elements = input.getNumElements();

        const void *input_data = input.getDataPointer();
        void *output_data = output->getDataPointer();

        dim3 gridSize((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(sigmoid_kernel<float>, gridSize, blockSize, 0, 0,
                                                    static_cast<const float *>(input_data),
                                                    static_cast<float *>(output_data),
                                                    num_elements));
            break;
        case TensorDataType::FLOAT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(sigmoid_kernel<double>, gridSize, blockSize, 0, 0,
                                                    static_cast<const double *>(input_data),
                                                    static_cast<double *>(output_data),
                                                    num_elements));
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
