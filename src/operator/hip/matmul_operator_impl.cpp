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
        // Shared memory tiles for A and B
        __shared__ T tile_A[TILE_SIZE][TILE_SIZE];
        __shared__ T tile_B[TILE_SIZE][TILE_SIZE];

        // Row and column index for the output matrix
        size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
        size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

        T sum = 0;

        // Loop over all the tiles needed to compute the result
        for (size_t t = 0; t < (dim_A_col + TILE_SIZE - 1) / TILE_SIZE; ++t)
        {
            // Load data into shared memory
            if (row < dim_A_row && t * TILE_SIZE + threadIdx.x < dim_A_col)
            {
                tile_A[threadIdx.y][threadIdx.x] = A[row * dim_A_col + t * TILE_SIZE + threadIdx.x];
            }
            else // out of bounds
            {
                tile_A[threadIdx.y][threadIdx.x] = 0;
            }

            if (col < dim_B_col && t * TILE_SIZE + threadIdx.y < dim_A_col)
            {
                tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * dim_B_col + col];
            }
            else // out of bounds
            {
                tile_B[threadIdx.y][threadIdx.x] = 0;
            }

            __syncthreads(); // Synchronize to ensure all threads have loaded data

            // Perform multiplication for this tile
            for (size_t k = 0; k < TILE_SIZE; ++k)
            {
                sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
            }

            __syncthreads(); // Synchronize before loading the next tile
        }

        // Write the result to the output matrix
        if (row < dim_A_row && col < dim_B_col)
        {
            C[row * dim_B_col + col] = sum;
        }
    }

    OperatorExecuteResult MatMulOperatorImpl::execute(const std::vector<Tensor> &inputs,
                                                      std::vector<Tensor *> &outputs,
                                                      const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        if (inputs.size() != 2)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        if (outputs.size() != 1 || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const Tensor &A = inputs[0];
        const Tensor &B = inputs[1];
        Tensor *Y = outputs[0];

        const std::vector<size_t> &shape_A = A.getDims();
        const std::vector<size_t> &shape_B = B.getDims();

        if (shape_A.size() != 2 || shape_B.size() != 2)
        {
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }

        size_t dim_A_row = shape_A[0];
        size_t dim_A_col = shape_A[1];
        size_t dim_B_row = shape_B[0];
        size_t dim_B_col = shape_B[1];

        if (dim_A_col != dim_B_row)
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        // Reshape output tensor
        std::vector<size_t> output_shape = {dim_A_row, dim_B_col};
        Y->reshape(output_shape);
        Y->setDataType(A.getDataType());

        // Allocate buffers
        const void *A_data = A.getBuffer()->getDataPointer();
        const void *B_data = B.getBuffer()->getDataPointer();
        void *Y_data = Y->getBuffer()->getDataPointer();

        // Kernel launch configuration
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((dim_B_col + TILE_SIZE - 1) / TILE_SIZE, (dim_A_row + TILE_SIZE - 1) / TILE_SIZE);

        // Launch the kernel based on the data type
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
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        // Synchronize the device to wait for the kernel to finish
        hipErrorCheck(hipDeviceSynchronize());

        return OperatorExecuteResult::SUCCESS;
    }
}

#endif
