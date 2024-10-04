#ifdef USE_HIP
#include "operator/operators.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#define TILE_SIZE 16

namespace HIP_OP
{

    template <typename T>
    __global__ void matmul_kernel(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
                                  size_t dim_A_row, size_t dim_A_col, size_t dim_B_col)
    {

        __shared__ T tile_A[TILE_SIZE][TILE_SIZE];
        __shared__ T tile_B[TILE_SIZE][TILE_SIZE];

        size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
        size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

        T sum = 0;

        for (size_t t = 0; t < (dim_A_col + TILE_SIZE - 1) / TILE_SIZE; ++t)
        {

            if (row < dim_A_row && t * TILE_SIZE + threadIdx.x < dim_A_col)
            {
                tile_A[threadIdx.y][threadIdx.x] = A[row * dim_A_col + t * TILE_SIZE + threadIdx.x];
            }
            else
            {
                tile_A[threadIdx.y][threadIdx.x] = 0;
            }

            if (col < dim_B_col && t * TILE_SIZE + threadIdx.y < dim_A_col)
            {
                tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * dim_B_col + col];
            }
            else
            {
                tile_B[threadIdx.y][threadIdx.x] = 0;
            }

            __syncthreads();

            for (size_t k = 0; k < TILE_SIZE; ++k)
            {
                sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < dim_A_row && col < dim_B_col)
        {
            C[row * dim_B_col + col] = sum;
        }
    }

    OperatorExecuteResult MatMulOperatorImpl::execute(const std::vector<Tensor> &inputs,
                                                      std::vector<Tensor *> &outputs,
                                                      const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        const Tensor &A = inputs[0];
        const Tensor &B = inputs[1];
        Tensor *Y = outputs[0];

        const std::vector<size_t> &shape_A = A.getDims();
        const std::vector<size_t> &shape_B = B.getDims();
        size_t dim_A_row = shape_A[0];
        size_t dim_A_col = shape_A[1];
        size_t dim_B_row = shape_B[0];
        size_t dim_B_col = shape_B[1];

        const void *A_data = A.getDataPointer();
        const void *B_data = B.getDataPointer();
        void *Y_data = Y->getDataPointer();

        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize(CeilDiv(dim_B_col, TILE_SIZE), CeilDiv(dim_A_row, TILE_SIZE));

        switch (A.getDataType())
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(matmul_kernel<float>, gridSize, blockSize, 0, 0,
                                                    static_cast<const float *>(A_data), static_cast<const float *>(B_data),
                                                    static_cast<float *>(Y_data), dim_A_row, dim_A_col, dim_B_col));
            break;
        case TensorDataType::FLOAT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(matmul_kernel<double>, gridSize, blockSize, 0, 0,
                                                    static_cast<const double *>(A_data), static_cast<const double *>(B_data),
                                                    static_cast<double *>(Y_data), dim_A_row, dim_A_col, dim_B_col));
            break;
        case TensorDataType::INT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(matmul_kernel<int32_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int32_t *>(A_data), static_cast<const int32_t *>(B_data),
                                                    static_cast<int32_t *>(Y_data), dim_A_row, dim_A_col, dim_B_col));
            break;
        case TensorDataType::INT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(matmul_kernel<int64_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int64_t *>(A_data), static_cast<const int64_t *>(B_data),
                                                    static_cast<int64_t *>(Y_data), dim_A_row, dim_A_col, dim_B_col));
            break;
        case TensorDataType::INT8:
            hipKernelLaunchCheck(hipLaunchKernelGGL(matmul_kernel<int8_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int8_t *>(A_data), static_cast<const int8_t *>(B_data),
                                                    static_cast<int8_t *>(Y_data), dim_A_row, dim_A_col, dim_B_col));
            break;
        case TensorDataType::UINT8:
            hipKernelLaunchCheck(hipLaunchKernelGGL(matmul_kernel<uint8_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const uint8_t *>(A_data), static_cast<const uint8_t *>(B_data),
                                                    static_cast<uint8_t *>(Y_data), dim_A_row, dim_A_col, dim_B_col));
            break;

        case TensorDataType::FLOAT16:
            hipKernelLaunchCheck(hipLaunchKernelGGL(matmul_kernel<__half>, gridSize, blockSize, 0, 0,
                                                    static_cast<const __half *>(A_data), static_cast<const __half *>(B_data),
                                                    static_cast<__half *>(Y_data), dim_A_row, dim_A_col, dim_B_col));
            break;
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        hipErrorCheck(hipDeviceSynchronize());

        return OperatorExecuteResult::SUCCESS;
    }
}

#endif
