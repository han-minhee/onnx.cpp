#ifdef USE_HIP
#include "operator/operators.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <numeric>
#include "types/half_t.hpp"

#define BLOCK_SIZE 256
#define MAX_DIMS 8

namespace HIP_OP
{
    // Specialized kernel for __half
    __global__ void transpose_kernel_half(const __half *__restrict__ input_data, const size_t *__restrict__ input_dims,
                                          const size_t *__restrict__ input_strides, __half *__restrict__ output_data,
                                          const size_t *__restrict__ output_strides, const size_t *__restrict__ perm,
                                          size_t num_elements, size_t rank)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;

        size_t remaining = idx;
        size_t input_indices[MAX_DIMS];
        for (int i = rank - 1; i >= 0; --i)
        {
            input_indices[i] = remaining % input_dims[i];
            remaining /= input_dims[i];
        }

        size_t output_linear_idx = 0;
        for (int i = 0; i < rank; ++i)
        {
            output_linear_idx += input_indices[perm[i]] * output_strides[i];
        }

        output_data[output_linear_idx] = input_data[idx];
    }

    // Template for general types
    template <typename T>
    __global__ void transpose_kernel(const T *__restrict__ input_data, const size_t *__restrict__ input_dims,
                                     const size_t *__restrict__ input_strides, T *__restrict__ output_data,
                                     const size_t *__restrict__ output_strides, const size_t *__restrict__ perm,
                                     size_t num_elements, size_t rank)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;

        size_t remaining = idx;
        size_t input_indices[MAX_DIMS];
        for (int i = rank - 1; i >= 0; --i)
        {
            input_indices[i] = remaining % input_dims[i];
            remaining /= input_dims[i];
        }

        size_t output_linear_idx = 0;
        for (int i = 0; i < rank; ++i)
        {
            output_linear_idx += input_indices[perm[i]] * output_strides[i];
        }

        output_data[output_linear_idx] = input_data[idx];
    }

    // Specialized executeTranspose for __half
    OperatorExecuteResult executeTranspose_half(const Tensor &input_tensor, Tensor *output_tensor, const std::vector<size_t> &perm, size_t num_elements, size_t rank)
    {
        const void *input_data = input_tensor.getDataPointer();
        void *output_data = output_tensor->getDataPointer();

        // Allocate memory for permutation on the device
        size_t *d_perm;
        hipErrorCheck(hipMalloc(&d_perm, rank * sizeof(size_t)));
        hipErrorCheck(hipMemcpy(d_perm, perm.data(), rank * sizeof(size_t), hipMemcpyHostToDevice));

        dim3 gridSize((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        const __half *input_data_half = static_cast<const __half *>(input_data);
        __half *output_data_half = static_cast<__half *>(output_data);

        // Launch the specialized __half kernel
        hipKernelLaunchCheck(hipLaunchKernelGGL(transpose_kernel_half, gridSize, blockSize, 0, 0,
                                                input_data_half, input_tensor.d_getDims(), input_tensor.d_getStrides(),
                                                output_data_half, output_tensor->d_getStrides(), d_perm,
                                                num_elements, rank));

        hipErrorCheck(hipFree(d_perm));

        return OperatorExecuteResult::SUCCESS;
    }

    // Template for general executeTranspose
    template <typename T>
    OperatorExecuteResult executeTranspose(const Tensor &input_tensor, Tensor *output_tensor, const std::vector<size_t> &perm, size_t num_elements, size_t rank)
    {
        const T *input_data = input_tensor.data<T>();
        T *output_data = output_tensor->data<T>();

        // Allocate memory for permutation on the device
        size_t *d_perm;
        hipErrorCheck(hipMalloc(&d_perm, rank * sizeof(size_t)));
        hipErrorCheck(hipMemcpy(d_perm, perm.data(), rank * sizeof(size_t), hipMemcpyHostToDevice));

        dim3 gridSize((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        // Launch the general templated kernel
        hipKernelLaunchCheck(hipLaunchKernelGGL(transpose_kernel<T>, gridSize, blockSize, 0, 0,
                                                input_data, input_tensor.d_getDims(), input_tensor.d_getStrides(),
                                                output_data, output_tensor->d_getStrides(), d_perm,
                                                num_elements, rank));

        hipErrorCheck(hipFree(d_perm));

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult TransposeOperatorImpl::execute(const std::vector<Tensor> &inputs,
                                                         std::vector<Tensor *> &outputs,
                                                         const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        const Tensor &input_tensor = inputs[0];
        Tensor *output_tensor = outputs[0];

        const auto &input_dims = input_tensor.getDims();
        const auto &output_dims = output_tensor->getDims();

        size_t rank = input_dims.size();
        size_t num_elements = input_tensor.getNumElements();

        std::vector<size_t> perm(rank);
        if (attributes.count("perm"))
        {
            const auto &perm_attr = std::get<std::vector<int64_t>>(attributes.at("perm"));
            if (perm_attr.size() != rank)
                return OperatorExecuteResult::ATTRIBUTE_ERROR;
            std::transform(perm_attr.begin(), perm_attr.end(), perm.begin(), [](int64_t val)
                           { return static_cast<size_t>(val); });
        }
        else
        {
            std::iota(perm.rbegin(), perm.rend(), 0);
        }

        switch (input_tensor.getDataType())
        {
        case TensorDataType::FLOAT32:
            return executeTranspose<float>(input_tensor, output_tensor, perm, num_elements, rank);
        case TensorDataType::FLOAT64:
            return executeTranspose<double>(input_tensor, output_tensor, perm, num_elements, rank);
        case TensorDataType::INT32:
            return executeTranspose<int32_t>(input_tensor, output_tensor, perm, num_elements, rank);
        case TensorDataType::INT64:
            return executeTranspose<int64_t>(input_tensor, output_tensor, perm, num_elements, rank);
        case TensorDataType::INT8:
            return executeTranspose<int8_t>(input_tensor, output_tensor, perm, num_elements, rank);
        case TensorDataType::UINT8:
            return executeTranspose<uint8_t>(input_tensor, output_tensor, perm, num_elements, rank);
        case TensorDataType::FLOAT16:
            return executeTranspose_half(input_tensor, output_tensor, perm, num_elements, rank);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
}

#endif
