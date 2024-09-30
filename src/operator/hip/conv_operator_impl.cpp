#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#include "utils.hpp"

#define MAX_DIMS 8
#define BLOCK_SIZE 256

namespace HIP_OP
{

    template <typename T>
    __global__ void conv_kernel(const T *input_data, const T *weight_data, const T *bias_data, T *output_data,
                                int64_t group, size_t N, size_t C, size_t M, size_t H_out, size_t W_out,
                                size_t kH, size_t kW, size_t H, size_t W_in,
                                size_t padH, size_t padW, size_t strideH, size_t strideW,
                                size_t dilationH, size_t dilationW)
    {
        // Calculate global indices
        int w_out = blockIdx.x * blockDim.x + threadIdx.x; // output width index
        int h_out = blockIdx.y * blockDim.y + threadIdx.y; // output height index
        int m = blockIdx.z % (M / group);                  // output channel within the group
        int n = blockIdx.z / (M / group);                  // batch index

        // Boundary check
        if (n >= N || m >= (M / group) || h_out >= H_out || w_out >= W_out)
            return;

        int g = blockIdx.z / (M / group); // Compute group index

        // Initialize the sum
        T sum = 0;

        // Perform convolution for this output element
        for (size_t c = 0; c < C / group; ++c) // Input channels per group
        {
            for (size_t kh = 0; kh < kH; ++kh) // Kernel height
            {
                for (size_t kw = 0; kw < kW; ++kw) // Kernel width
                {
                    int h_in = h_out * strideH + kh * dilationH - padH;
                    int w_in = w_out * strideW + kw * dilationW - padW;

                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W_in) // Input boundary check
                    {
                        size_t input_idx = n * (C * H * W_in) + (g * C / group + c) * (H * W_in) + h_in * W_in + w_in;
                        size_t weight_idx = (g * M / group + m) * (C / group * kH * kW) + c * (kH * kW) + kh * kW + kw;
                        sum = sum + input_data[input_idx] * weight_data[weight_idx];
                    }
                }
            }
        }

        // Add bias if available
        if (bias_data)
        {
            sum = sum + bias_data[g * M / group + m];
        }

        // Compute the output index
        size_t output_idx = n * (M * H_out * W_out) + (g * M / group + m) * (H_out * W_out) + h_out * W_out + w_out;
        output_data[output_idx] = sum;
    }

    template <typename T>
    OperatorExecuteResult executeConv(const Tensor &X, const Tensor &W, const Tensor *B, Tensor *Y,
                                      const std::vector<int64_t> &pads, const std::vector<int64_t> &strides,
                                      const std::vector<int64_t> &dilations, int64_t group, size_t N, size_t C,
                                      size_t H, size_t W_in, size_t M, size_t kH, size_t kW, size_t H_out, size_t W_out)
    {
        const T *input_data = X.data<T>();
        const T *weight_data = W.data<T>();

        const T *bias_data = B ? B->data<T>() : nullptr;

        T *output_data = Y->data<T>();

        // Zero-initialize output data
        hipErrorCheck(hipMemset(output_data, 0, N * M * H_out * W_out * sizeof(T)));

        // Calculate grid and block dimensions
        // Set the block dimensions (typically 16x16 threads per block for 2D data)
        dim3 blockDim(16, 16);

        // Calculate the number of blocks needed in the X and Y dimensions
        dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x,
                     (H_out + blockDim.y - 1) / blockDim.y,
                     N * (M / group)); // Combining N and M/group in the Z dimension

        // Launch the kernel
        hipKernelLaunchCheck(hipLaunchKernelGGL(conv_kernel<T>, gridDim, blockDim, 0, 0,
                                                input_data, weight_data, bias_data, output_data,
                                                group, N, C, M, H_out, W_out, kH, kW, H, W_in,
                                                pads[0], pads[1],
                                                strides[0], strides[1],
                                                dilations[0], dilations[1]));

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult ConvOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                    const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        const Tensor &X = inputs.at(0);
        const Tensor &W = inputs.at(1);

        bool has_bias = inputs.size() == 3;
        const Tensor *B = has_bias ? &inputs.at(2) : nullptr;

        if (outputs.empty() || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const std::vector<size_t> &X_dims = X.getDims(); // (N, C, H, W)
        const std::vector<size_t> &W_dims = W.getDims(); // (M, C/group, kH, kW)

        // Check dimensions
        if (X_dims.size() != 4 || W_dims.size() != 4)
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR; // supports 4D tensors
        }

        // Get attributes with defaults
        std::string auto_pad = "NOTSET";
        std::vector<int64_t> dilations = {1, 1};
        int64_t group = 1;
        std::vector<int64_t> kernel_shape = {static_cast<int>(W_dims[2]), static_cast<int>(W_dims[3])}; // (kH, kW)
        std::vector<int64_t> pads = {0, 0, 0, 0};                                                       // (padH_begin, padW_begin, padH_end, padW_end)
        std::vector<int64_t> strides = {1, 1};

        for (const auto &[key, value] : attributes)
        {
            if (key == "auto_pad")
                auto_pad = std::get<std::string>(value);
            else if (key == "dilations")
                dilations = std::get<std::vector<int64_t>>(value);
            else if (key == "group")
                group = std::get<int64_t>(value);
            else if (key == "kernel_shape")
                kernel_shape = std::get<std::vector<int64_t>>(value);
            else if (key == "pads")
                pads = std::get<std::vector<int64_t>>(value);
            else if (key == "strides")
                strides = std::get<std::vector<int64_t>>(value);
        }

        size_t N = X_dims[0], C = X_dims[1], H = X_dims[2], W_in = X_dims[3];
        size_t M = W_dims[0], kH = kernel_shape[0], kW = kernel_shape[1];

        if (kH > H + pads[0] + pads[2] || kW > W_in + pads[1] + pads[3])
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR; // Invalid kernel size
        }

        if (dilations.size() != 2 || pads.size() != 4 || strides.size() != 2)
        {
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        }

        size_t H_out = static_cast<size_t>(std::floor((H + pads[0] + pads[2] - (kH - 1) * dilations[0] - 1) / static_cast<size_t>(strides[0]) + 1));
        size_t W_out = static_cast<size_t>(std::floor((W_in + pads[1] + pads[3] - (kW - 1) * dilations[1] - 1) / static_cast<size_t>(strides[1]) + 1));

        Tensor *Y = outputs[0];

        switch (X.getDataType())
        {
        case TensorDataType::FLOAT32:
            return executeConv<float>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        case TensorDataType::FLOAT64:
            return executeConv<double>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        case TensorDataType::INT32:
            return executeConv<int32_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        case TensorDataType::INT64:
            return executeConv<int64_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        case TensorDataType::INT8:
            return executeConv<int8_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        case TensorDataType::UINT8:
            return executeConv<uint8_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        case TensorDataType::FLOAT16:
            return executeConv<half_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
};

#endif
