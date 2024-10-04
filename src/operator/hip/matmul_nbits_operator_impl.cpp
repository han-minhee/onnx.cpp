#ifdef USE_HIP
#include "operator/operators.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

namespace HIP_OP
{
    /// FIXME: Fuse dequantize and matmul into a single kernel

    template <typename T>
    __global__ void dequantize_4bit_kernel(const uint8_t *B_quant, const float *scales, T *B_dequant,
                                           size_t N, size_t K, size_t block_size)
    {
        /// FIXME: Hardcoded zero point
        const float zero_point = 8.0f;
        size_t n_blocks_per_col = K / block_size;

        size_t n = blockIdx.x * blockDim.x + threadIdx.x;
        size_t k = blockIdx.y * blockDim.y + threadIdx.y;

        if (n < N && k < K)
        {
            // printf("n: %d, k: %d\n", n, k);
            size_t block = k / block_size;
            size_t block_offset = k % block_size;
            size_t quant_idx = (n * n_blocks_per_col + block) * (block_size / 2) + block_offset / 2;

            if (quant_idx >= (N * K / 2))
                return;

            uint8_t quant_value = B_quant[quant_idx];
            uint8_t bit_value = (block_offset % 2 == 0) ? (quant_value & 0x0F) : ((quant_value >> 4) & 0x0F);
            float scale = scales[n * n_blocks_per_col + block];
            B_dequant[n * K + k] = static_cast<T>((bit_value - zero_point) * scale);
        }
    }

    template <typename T>
    __global__ void matmul_nbits_kernel(const T *__restrict__ A, const T *__restrict__ B_dequant, T *__restrict__ C,
                                        size_t batch_size, size_t M, size_t N, size_t K)
    {
        size_t b = blockIdx.z;
        size_t row = blockIdx.y * blockDim.y + threadIdx.y;
        size_t col = blockIdx.x * blockDim.x + threadIdx.x;

        if (b < batch_size && row < M && col < N)
        {
            T sum = 0;
            for (size_t k = 0; k < K; ++k)
            {
                sum += A[b * M * K + row * K + k] * B_dequant[col * K + k];
            }
            C[b * M * N + row * N + col] = sum;
        }
    }

    OperatorExecuteResult MatMulNBitsOperatorImpl::execute(const std::vector<Tensor> &inputs,
                                                           std::vector<Tensor *> &outputs,
                                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes,
                                                           Device *device)
    {
        const Tensor &A = inputs[0];
        const Tensor &B_quant = inputs[1];
        const Tensor &scales = inputs[2];
        Tensor *Y = outputs[0];

        const std::vector<size_t> &shape_A = A.getDims();
        const std::vector<size_t> &shape_B_quant = B_quant.getDims();

        size_t batch_size = shape_A[0];
        size_t M = shape_A[1];
        size_t K = shape_A[2];
        size_t N = shape_B_quant[0];

        size_t block_size = 32;
        if (attributes.count("block_size") > 0)
        {
            block_size = std::get<int64_t>(attributes.at("block_size"));
        }

        const void *A_data = A.getDataPointer();
        const void *B_quant_data = B_quant.getDataPointer();
        const void *scales_data = scales.getDataPointer();
        void *Y_data = Y->getDataPointer();

        void *d_B_dequant;
        hipErrorCheck(hipMalloc(&d_B_dequant, N * K * sizeof(float)));

        dim3 deq_block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 deq_grid_size(CeilDiv(N, BLOCK_SIZE_X), CeilDiv(K, BLOCK_SIZE_Y));

        dim3 matmul_block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 matmul_grid_size(CeilDiv(N, BLOCK_SIZE_X), CeilDiv(M, BLOCK_SIZE_Y), batch_size);

        switch (A.getDataType())
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(dequantize_4bit_kernel<float>, deq_grid_size, deq_block_size, 0, 0,
                                                    static_cast<const uint8_t *>(B_quant_data),
                                                    static_cast<const float *>(scales_data),
                                                    static_cast<float *>(d_B_dequant),
                                                    N, K, block_size));

            hipKernelLaunchCheck(hipLaunchKernelGGL(matmul_nbits_kernel<float>, matmul_grid_size, matmul_block_size, 0, 0,
                                                    static_cast<const float *>(A_data),
                                                    static_cast<float *>(d_B_dequant),
                                                    static_cast<float *>(Y_data),
                                                    batch_size, M, N, K));

            hipErrorCheck(hipFree(d_B_dequant));
            return OperatorExecuteResult::SUCCESS;

        default:
            hipErrorCheck(hipFree(d_B_dequant));
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
}

#endif