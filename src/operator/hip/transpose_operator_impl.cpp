#ifdef USE_HIP
#include "operator/operators.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <numeric>

#define BLOCK_SIZE 256
#define MAX_DIMS 8

namespace HIP_OP
{

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

    OperatorExecuteResult TransposeOperatorImpl::execute(const std::vector<Tensor> &inputs,
                                                         std::vector<Tensor *> &outputs,
                                                         const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        const Tensor &input_tensor = inputs[0];
        Tensor *output_tensor = outputs[0];

        const auto &input_dims = input_tensor.getDims();
        const auto &output_dims = output_tensor->getDims();
        const auto &input_strides = input_tensor.getStrides();
        const auto &output_strides = output_tensor->getStrides();

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

        const void *input_data = input_tensor.getBuffer()->getDataPointer();
        void *output_data = output_tensor->getBuffer()->getDataPointer();

        size_t *d_input_dims = input_tensor.d_getDims();
        size_t *d_output_strides = output_tensor->d_getStrides();
        size_t *d_perm;
        hipErrorCheck(hipMalloc(&d_perm, rank * sizeof(size_t)));
        hipErrorCheck(hipMemcpy(d_perm, perm.data(), rank * sizeof(size_t), hipMemcpyHostToDevice));

        dim3 gridSize((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        switch (input_tensor.getDataType())
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(transpose_kernel<float>, gridSize, blockSize, 0, 0,
                                                    static_cast<const float *>(input_data), d_input_dims, input_tensor.d_getStrides(),
                                                    static_cast<float *>(output_data), d_output_strides, d_perm,
                                                    num_elements, rank));
            break;
        case TensorDataType::FLOAT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(transpose_kernel<double>, gridSize, blockSize, 0, 0,
                                                    static_cast<const double *>(input_data), d_input_dims, input_tensor.d_getStrides(),
                                                    static_cast<double *>(output_data), d_output_strides, d_perm,
                                                    num_elements, rank));
            break;
        case TensorDataType::INT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(transpose_kernel<int32_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int32_t *>(input_data), d_input_dims, input_tensor.d_getStrides(),
                                                    static_cast<int32_t *>(output_data), d_output_strides, d_perm,
                                                    num_elements, rank));
            break;
        case TensorDataType::INT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(transpose_kernel<int64_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int64_t *>(input_data), d_input_dims, input_tensor.d_getStrides(),
                                                    static_cast<int64_t *>(output_data), d_output_strides, d_perm,
                                                    num_elements, rank));
            break;
        case TensorDataType::INT8:
            hipKernelLaunchCheck(hipLaunchKernelGGL(transpose_kernel<int8_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int8_t *>(input_data), d_input_dims, input_tensor.d_getStrides(),
                                                    static_cast<int8_t *>(output_data), d_output_strides, d_perm,
                                                    num_elements, rank));
            break;
        case TensorDataType::UINT8:
            hipKernelLaunchCheck(hipLaunchKernelGGL(transpose_kernel<uint8_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const uint8_t *>(input_data), d_input_dims, input_tensor.d_getStrides(),
                                                    static_cast<uint8_t *>(output_data), d_output_strides, d_perm,
                                                    num_elements, rank));
            break;
        default:
            hipErrorCheck(hipFree(d_perm));
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        hipErrorCheck(hipFree(d_perm));

        return OperatorExecuteResult::SUCCESS;
    }
}

#endif
