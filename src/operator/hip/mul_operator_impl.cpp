#ifdef USE_HIP
#include "operator/operators.hpp"
#include "operator/elementwise_operator.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#include "utils.hpp"

#define MAX_DIMS 8
#define BLOCK_SIZE 256

namespace HIP_OP
{

    template <typename T>
    __global__ void mul_kernel(const T *__restrict__ A_data, const size_t *__restrict__ A_dims, const size_t *__restrict__ A_strides,
                               const T *__restrict__ B_data, const size_t *__restrict__ B_dims, const size_t *__restrict__ B_strides,
                               T *__restrict__ C_data, const size_t *__restrict__ C_dims, const size_t *__restrict__ C_strides,
                               size_t num_elements, size_t A_ndims, size_t B_ndims, size_t C_ndims)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;

        size_t indices[MAX_DIMS];
        size_t tmp = idx;
        for (int i = C_ndims - 1; i >= 0; --i)
        {
            indices[i] = tmp % C_dims[i];
            tmp /= C_dims[i];
        }

        size_t A_idx = 0, B_idx = 0;
        for (int i = 0; i < C_ndims; ++i)
        {
            if (i < A_ndims)
                A_idx += (indices[i] % A_dims[i]) * A_strides[i];
            if (i < B_ndims)
                B_idx += (indices[C_ndims - B_ndims + i] % B_dims[i]) * B_strides[i];
        }

        C_data[idx] = A_data[A_idx] + B_data[B_idx];
    }
    OperatorExecuteResult MulOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        if (inputs.size() != 2)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }
        if (outputs.empty() || outputs.size() != 1)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const Tensor &A = inputs[0];
        const Tensor &B = inputs[1];
        Tensor *C = outputs[0];

        if (A.getDataType() != B.getDataType())
        {
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        TensorDataType dtype = A.getDataType();
        size_t num_elements_C = C->getNumElements();

        const void *A_data = A.getBuffer()->getDataPointer();
        const void *B_data = B.getBuffer()->getDataPointer();
        void *C_data = C->getBuffer()->getDataPointer();

        const auto &A_dims = A.getDims();
        const auto &B_dims = B.getDims();
        const auto &C_dims = C->getDims();
        const auto &A_strides = A.getStrides();
        const auto &B_strides = B.getStrides();
        const auto &C_strides = C->getStrides();

        size_t A_ndims = A_dims.size();
        size_t B_ndims = B_dims.size();
        size_t C_ndims = C_dims.size();

        size_t *d_A_dims, *d_B_dims, *d_C_dims;
        size_t *d_A_strides, *d_B_strides, *d_C_strides;

        hipErrorCheck(hipMalloc(&d_A_dims, A_ndims * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_B_dims, B_ndims * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_C_dims, C_ndims * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_A_strides, A_ndims * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_B_strides, B_ndims * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_C_strides, C_ndims * sizeof(size_t)));

        hipErrorCheck(hipMemcpy(d_A_dims, A_dims.data(), A_ndims * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_B_dims, B_dims.data(), B_ndims * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_C_dims, C_dims.data(), C_ndims * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_A_strides, A_strides.data(), A_ndims * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_B_strides, B_strides.data(), B_ndims * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_C_strides, C_strides.data(), C_ndims * sizeof(size_t), hipMemcpyHostToDevice));

        dim3 gridSize((num_elements_C + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(mul_kernel<float>, gridSize, blockSize, 0, 0,
                                                    static_cast<const float *>(A_data), d_A_dims, d_A_strides,
                                                    static_cast<const float *>(B_data), d_B_dims, d_B_strides,
                                                    static_cast<float *>(C_data), d_C_dims, d_C_strides,
                                                    num_elements_C, A_ndims, B_ndims, C_ndims));
            break;
        // Add cases for other data types as needed
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        hipErrorCheck(hipDeviceSynchronize());

        // Clean up
        hipErrorCheck(hipFree(d_A_dims));
        hipErrorCheck(hipFree(d_B_dims));
        hipErrorCheck(hipFree(d_C_dims));
        hipErrorCheck(hipFree(d_A_strides));
        hipErrorCheck(hipFree(d_B_strides));
        hipErrorCheck(hipFree(d_C_strides));

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif