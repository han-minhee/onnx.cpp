#include "operator/operators.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <vector>
#include <cstring>

namespace CPU_OP
{
    void dequantize_4bit(const uint8_t *B_quant, const float *scales, std::vector<float> &B_dequant, size_t N, size_t K, size_t block_size)
    {
        const float zero_point = 8.0f;
        size_t n_blocks_per_col = K / block_size;

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t k = 0; k < K; ++k)
            {
                size_t block = k / block_size;
                size_t block_offset = k % block_size;
                size_t quant_idx = (n * n_blocks_per_col + block) * (block_size / 2) + block_offset / 2;
                uint8_t quant_value = B_quant[quant_idx];
                uint8_t bit_value = (block_offset % 2 == 0) ? (quant_value & 0x0F) : ((quant_value >> 4) & 0x0F);
                float scale = scales[n * n_blocks_per_col + block];
                B_dequant[n * K + k] = (bit_value * scale) - (scale * zero_point);
            }
        }
    }

    OperatorExecuteResult executeMatMulNBits(const Tensor &A, const Tensor &B_quant, const Tensor &scales, const Tensor *zero_points_tensor, Tensor *Y, const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        std::vector<size_t> A_shape = A.getDims();
        const size_t batch_size = A_shape[0];
        const size_t M = A_shape[1];
        const size_t K = A_shape[2];

        std::vector<size_t> B_quant_shape = B_quant.getDims();
        const size_t N = B_quant_shape[0];

        const uint8_t *B_quant_data = B_quant.data<uint8_t>();
        const float *scales_data = scales.data<float>();

        size_t block_size = 32;
        if (attributes.count("block_size") > 0)
        {
            block_size = std::get<int64_t>(attributes.at("block_size"));
        }

        std::vector<float> B_dequant(N * K, 0.0f);
        dequantize_4bit(B_quant_data, scales_data, B_dequant, N, K, block_size);

        const float *A_data = A.data<float>();
        float *Y_data = Y->data<float>();

        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t m = 0; m < M; ++m)
            {
                for (size_t n = 0; n < N; ++n)
                {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k)
                    {
                        sum += A_data[b * M * K + m * K + k] * B_dequant[n * K + k];
                    }
                    Y_data[b * M * N + m * N + n] = sum;
                }
            }
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult MatMulNBitsOperatorImpl::execute(
        const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
        const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        const Tensor &A = inputs.at(0);
        const Tensor &B_quant = inputs.at(1);
        const Tensor &scales = inputs.at(2);
        const Tensor *zero_points_tensor = nullptr;

        if (inputs.size() > 3)
        {
            zero_points_tensor = &inputs.at(3);
        }

        Tensor *Y = outputs[0];

        switch (A.getDataType())
        {
        case TensorDataType::FLOAT32:
            return executeMatMulNBits(A, B_quant, scales, zero_points_tensor, Y, attributes);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
}