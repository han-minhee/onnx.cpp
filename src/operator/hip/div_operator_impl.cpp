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
    __global__ void div_kernel(const T *__restrict__ A_data, const size_t *__restrict__ A_dims, const size_t *__restrict__ A_strides,
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

        C_data[idx] = A_data[A_idx] / B_data[B_idx];
    }
    OperatorExecuteResult DivOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {

        const Tensor &A = inputs[0];
        const Tensor &B = inputs[1];
        Tensor *C = outputs[0];

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

        size_t *d_A_dims = A.d_getDims();
        size_t *d_B_dims = B.d_getDims();
        size_t *d_C_dims = C->d_getDims();

        size_t *d_A_strides = A.d_getStrides();
        size_t *d_B_strides = B.d_getStrides();
        size_t *d_C_strides = C->d_getStrides();

        dim3 gridSize((num_elements_C + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(div_kernel<float>, gridSize, blockSize, 0, 0,
                                                    static_cast<const float *>(A_data), d_A_dims, d_A_strides,
                                                    static_cast<const float *>(B_data), d_B_dims, d_B_strides,
                                                    static_cast<float *>(C_data), d_C_dims, d_C_strides,
                                                    num_elements_C, A_ndims, B_ndims, C_ndims));
            break;
        case TensorDataType::FLOAT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(div_kernel<double>, gridSize, blockSize, 0, 0,
                                                    static_cast<const double *>(A_data), d_A_dims, d_A_strides,
                                                    static_cast<const double *>(B_data), d_B_dims, d_B_strides,
                                                    static_cast<double *>(C_data), d_C_dims, d_C_strides,
                                                    num_elements_C, A_ndims, B_ndims, C_ndims));
            break;
        case TensorDataType::INT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(div_kernel<int64_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int64_t *>(A_data), d_A_dims, d_A_strides,
                                                    static_cast<const int64_t *>(B_data), d_B_dims, d_B_strides,
                                                    static_cast<int64_t *>(C_data), d_C_dims, d_C_strides,
                                                    num_elements_C, A_ndims, B_ndims, C_ndims));
            break;
        case TensorDataType::INT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(div_kernel<int32_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int32_t *>(A_data), d_A_dims, d_A_strides,
                                                    static_cast<const int32_t *>(B_data), d_B_dims, d_B_strides,
                                                    static_cast<int32_t *>(C_data), d_C_dims, d_C_strides,
                                                    num_elements_C, A_ndims, B_ndims, C_ndims));
            break;
        
        case TensorDataType::FLOAT16:
            hipKernelLaunchCheck(hipLaunchKernelGGL(div_kernel<half_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const half_t *>(A_data), d_A_dims, d_A_strides,
                                                    static_cast<const half_t *>(B_data), d_B_dims, d_B_strides,
                                                    static_cast<half_t *>(C_data), d_C_dims, d_C_strides,
                                                    num_elements_C, A_ndims, B_ndims, C_ndims));
            break;

        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        hipErrorCheck(hipDeviceSynchronize());

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif