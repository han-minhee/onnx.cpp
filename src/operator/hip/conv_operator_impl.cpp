#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#include "utils.hpp"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

namespace HIP_OP
{

    template <typename T>
    __global__ void conv_kernel(const T *input_data, const T *weight_data, const T *bias_data, T *output_data,
                                int64_t group, size_t N, size_t C, size_t M, size_t H_out, size_t W_out,
                                size_t kH, size_t kW, size_t H, size_t W_in,
                                size_t padH, size_t padW, size_t strideH, size_t strideW,
                                size_t dilationH, size_t dilationW)
    {
        int w_out = blockIdx.x * blockDim.x + threadIdx.x;
        int h_out = blockIdx.y * blockDim.y + threadIdx.y;
        int m = blockIdx.z % (M / group);
        int n = blockIdx.z / (M / group);

        if (n >= N || m >= (M / group) || h_out >= H_out || w_out >= W_out)
            return;

        T sum = 0;

        for (size_t c = 0; c < C / group; ++c)
        {
            for (size_t kh = 0; kh < kH; ++kh)
            {
                for (size_t kw = 0; kw < kW; ++kw)
                {
                    int h_in = h_out * strideH + kh * dilationH - padH;
                    int w_in = w_out * strideW + kw * dilationW - padW;

                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W_in)
                    {
                        size_t input_idx = n * (C * H * W_in) + (n * C / group + c) * (H * W_in) + h_in * W_in + w_in;
                        size_t weight_idx = (n * M / group + m) * (C / group * kH * kW) + c * (kH * kW) + kh * kW + kw;
                        sum = sum + input_data[input_idx] * weight_data[weight_idx];
                    }
                }
            }
        }

        if (bias_data)
        {
            sum = sum + bias_data[n * M / group + m];
        }

        size_t output_idx = n * (M * H_out * W_out) + (n * M / group + m) * (H_out * W_out) + h_out * W_out + w_out;
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

        hipErrorCheck(hipMemset(output_data, 0, N * M * H_out * W_out * sizeof(T)));

        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridDim(CeilDiv(W_out, BLOCK_SIZE_X),
                     CeilDiv(H_out, BLOCK_SIZE_Y),
                     N * M / group);

        hipKernelLaunchCheck(hipLaunchKernelGGL(conv_kernel<T>, gridDim, blockDim, 0, 0,
                                                input_data, weight_data, bias_data, output_data,
                                                group, N, C, M, H_out, W_out, kH, kW, H, W_in,
                                                pads[0], pads[1],
                                                strides[0], strides[1],
                                                dilations[0], dilations[1]));

        return OperatorExecuteResult::SUCCESS;
    }

    bool useWinograd(const std::vector<int64_t> &kernel_shape, const std::vector<int64_t> &strides,
                     const std::vector<int64_t> &dilations, size_t C, size_t M, size_t H, size_t W_in)
    {

        if (!(kernel_shape[0] == 3 && kernel_shape[1] == 3))
        {
            return false;
        }

        if (!(strides[0] == 1 && strides[1] == 1))
        {
            return false;
        }

        if (!(dilations[0] == 1 && dilations[1] == 1))
        {
            return false;
        }

        if (C < 16 || M < 16)
        {
            return false;
        }

        if (H < 16 || W_in < 16)
        {
            return false;
        }

        return true;
    }

    template <typename T>
    __device__ void winogradTransformInput(const T d[4][4], T transformed_input[4][4])
    {
        T temp[4][4];

        for (size_t i = 0; i < 4; ++i)
        {
            temp[i][0] = d[i][0] - d[i][2];
            temp[i][1] = d[i][1] + d[i][2];
            temp[i][2] = -d[i][1] + d[i][2];
            temp[i][3] = d[i][1] - d[i][3];
        }

        for (size_t i = 0; i < 4; ++i)
        {
            transformed_input[0][i] = temp[0][i] - temp[2][i];
            transformed_input[1][i] = temp[1][i] + temp[2][i];
            transformed_input[2][i] = -temp[1][i] + temp[2][i];
            transformed_input[3][i] = temp[1][i] - temp[3][i];
        }
    }

    template <typename T>
    __device__ void winogradTransformOutput(const T m[4][4], T output_tile[2][2])
    {
        T temp[2][4];

        for (size_t i = 0; i < 4; ++i)
        {
            temp[0][i] = m[0][i] + m[1][i] + m[2][i];
            temp[1][i] = m[1][i] - m[2][i] - m[3][i];
        }

        for (size_t i = 0; i < 2; ++i)
        {
            output_tile[i][0] = temp[i][0] + temp[i][1] + temp[i][2];
            output_tile[i][1] = temp[i][1] - temp[i][2] - temp[i][3];
        }
    }

    template <typename T>
    __global__ void winograd_conv_kernel(const T *input_data, const T *transformed_kernel_data, const T *bias_data, T *output_data,
                                         int64_t group, size_t N, size_t C, size_t M, size_t H_out, size_t W_out,
                                         size_t H, size_t W_in,
                                         size_t padH, size_t padW,
                                         size_t strideH, size_t strideW)
    {
        int w_out_tile = blockIdx.x * blockDim.x + threadIdx.x;
        int h_out_tile = blockIdx.y * blockDim.y + threadIdx.y;

        if (w_out_tile >= (W_out + 1) / 2 || h_out_tile >= (H_out + 1) / 2)
            return;

        int n = blockIdx.z / group;
        int g = blockIdx.z % group;

        if (n >= N || g >= group)
            return;

        for (size_t m = 0; m < M / group; ++m)
        {
            T output_tile[2][2] = {{0, 0}, {0, 0}};

            for (size_t c = 0; c < C / group; ++c)
            {
                T input_tile[4][4];

                for (size_t ih = 0; ih < 4; ++ih)
                {
                    for (size_t iw = 0; iw < 4; ++iw)
                    {
                        int h_in = (h_out_tile * 2) * strideH + ih - padH;
                        int w_in = (w_out_tile * 2) * strideW + iw - padW;

                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W_in)
                        {
                            size_t input_idx = n * (C * H * W_in) + (g * (C / group) + c) * (H * W_in) + h_in * W_in + w_in;
                            input_tile[ih][iw] = input_data[input_idx];
                        }
                        else
                        {
                            input_tile[ih][iw] = 0;
                        }
                    }
                }

                T transformed_input[4][4];
                winogradTransformInput<T>(input_tile, transformed_input);

                T transformed_kernel[4][4];
                size_t kernel_idx = ((g * (M / group) + m) * (C / group) + c) * 16;
                for (size_t i = 0; i < 4; ++i)
                {
                    for (size_t j = 0; j < 4; ++j)
                    {
                        transformed_kernel[i][j] = transformed_kernel_data[kernel_idx + i * 4 + j];
                    }
                }

                T transformed_output[4][4];
                for (size_t i = 0; i < 4; ++i)
                {
                    for (size_t j = 0; j < 4; ++j)
                    {
                        transformed_output[i][j] = transformed_input[i][j] * transformed_kernel[i][j];
                    }
                }

                T output_tile_partial[2][2];
                winogradTransformOutput<T>(transformed_output, output_tile_partial);

                for (size_t i = 0; i < 2; ++i)
                {
                    for (size_t j = 0; j < 2; ++j)
                    {
                        output_tile[i][j] += output_tile_partial[i][j];
                    }
                }
            }

            if (bias_data)
            {
                T bias = bias_data[g * (M / group) + m];
                for (size_t i = 0; i < 2; ++i)
                {
                    for (size_t j = 0; j < 2; ++j)
                    {
                        output_tile[i][j] += bias;
                    }
                }
            }

            for (size_t i = 0; i < 2; ++i)
            {
                for (size_t j = 0; j < 2; ++j)
                {
                    size_t h = h_out_tile * 2 + i;
                    size_t w = w_out_tile * 2 + j;

                    if (h < H_out && w < W_out)
                    {
                        size_t output_idx = n * (M * H_out * W_out) +
                                            (g * (M / group) + m) * (H_out * W_out) +
                                            h * W_out + w;
                        output_data[output_idx] = output_tile[i][j];
                    }
                }
            }
        }
    }

    template <typename T>
    void winogradTransformKernel(const T kernel[3][3], T transformed_kernel[4][4])
    {
        const T G[4][3] = {
            {1.0f, 0.0f, 0.0f},
            {0.5f, 0.5f, 0.5f},
            {0.5f, -0.5f, 0.5f},
            {0.0f, 0.0f, 1.0f}};
        const T GT[3][4] = {
            {1.0f, 0.5f, 0.5f, 0.0f},
            {0.0f, 0.5f, -0.5f, 0.0f},
            {0.0f, 0.5f, 0.5f, 1.0f}};

        T temp[4][3];

        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                temp[i][j] = 0;
                for (size_t k = 0; k < 3; ++k)
                {
                    temp[i][j] += G[i][k] * kernel[k][j];
                }
            }
        }

        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                transformed_kernel[i][j] = 0;
                for (size_t k = 0; k < 3; ++k)
                {
                    transformed_kernel[i][j] += temp[i][k] * GT[k][j];
                }
            }
        }
    }

    template <typename T>
    OperatorExecuteResult executeWinogradConv(const Tensor &X, const Tensor &W, const Tensor *B, Tensor *Y,
                                              const std::vector<int64_t> &pads, const std::vector<int64_t> &strides,
                                              const std::vector<int64_t> &dilations, int64_t group, size_t N, size_t C,
                                              size_t H, size_t W_in, size_t M, size_t kH, size_t kW, size_t H_out, size_t W_out)
    {
        if (kH != 3 || kW != 3)
        {
            return executeConv<T>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        }

        const T *input_data = X.data<T>();
        const T *weight_data = W.data<T>();
        const T *bias_data = B ? B->data<T>() : nullptr;

        T *output_data = Y->data<T>();

        hipErrorCheck(hipMemset(output_data, 0, N * M * H_out * W_out * sizeof(T)));

        std::vector<T> transformed_kernels(M * (C / group) * 16);
        for (size_t g = 0; g < static_cast<size_t>(group); ++g)
        {
            for (size_t m = 0; m < M / group; ++m)
            {
                for (size_t c = 0; c < C / group; ++c)
                {
                    T kernel[3][3];
                    for (size_t kh = 0; kh < 3; ++kh)
                    {
                        for (size_t kw = 0; kw < 3; ++kw)
                        {
                            size_t weight_idx = (g * (M / group) + m) * (C / group * kH * kW) + c * (kH * kW) + kh * kW + kw;
                            kernel[kh][kw] = weight_data[weight_idx];
                        }
                    }

                    T transformed_kernel[4][4];
                    winogradTransformKernel<T>(kernel, transformed_kernel);

                    size_t kernel_idx = ((g * (M / group) + m) * (C / group) + c) * 16;
                    for (size_t i = 0; i < 4; ++i)
                    {
                        for (size_t j = 0; j < 4; ++j)
                        {
                            transformed_kernels[kernel_idx + i * 4 + j] = transformed_kernel[i][j];
                        }
                    }
                }
            }
        }

        T *transformed_kernels_d;
        hipErrorCheck(hipMalloc(&transformed_kernels_d, transformed_kernels.size() * sizeof(T)));
        hipErrorCheck(hipMemcpy(transformed_kernels_d, transformed_kernels.data(), transformed_kernels.size() * sizeof(T), hipMemcpyHostToDevice));

        size_t tile_w = (W_out + 1) / 2;
        size_t tile_h = (H_out + 1) / 2;

        dim3 blockDim(16, 16);
        dim3 gridDim((tile_w + blockDim.x - 1) / blockDim.x,
                     (tile_h + blockDim.y - 1) / blockDim.y,
                     N * group);

        hipKernelLaunchCheck(hipLaunchKernelGGL(winograd_conv_kernel<T>, gridDim, blockDim, 0, 0,
                                                input_data, transformed_kernels_d, bias_data, output_data,
                                                group, N, C, M, H_out, W_out, H, W_in,
                                                pads[0], pads[1],
                                                strides[0], strides[1]));

        hipErrorCheck(hipFree(transformed_kernels_d));

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult ConvOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                    const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        const Tensor &X = inputs.at(0);
        const Tensor &W = inputs.at(1);
        bool has_bias = inputs.size() == 3;
        const Tensor *B = has_bias ? &inputs.at(2) : nullptr;
        const std::vector<size_t> &X_dims = X.getDims();
        const std::vector<size_t> &W_dims = W.getDims();

        std::string auto_pad = "NOTSET";
        std::vector<int64_t> dilations = {1, 1};
        int64_t group = 1;
        std::vector<int64_t> kernel_shape = {static_cast<int>(W_dims[2]), static_cast<int>(W_dims[3])};
        std::vector<int64_t> pads = {0, 0, 0, 0};
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

        size_t H_out = static_cast<size_t>(std::floor((H + pads[0] + pads[2] - (kH - 1) * dilations[0] - 1) / static_cast<size_t>(strides[0]) + 1));
        size_t W_out = static_cast<size_t>(std::floor((W_in + pads[1] + pads[3] - (kW - 1) * dilations[1] - 1) / static_cast<size_t>(strides[1]) + 1));

        Tensor *Y = outputs[0];

        bool use_winograd = useWinograd(kernel_shape, strides, dilations, C, M, H, W_in);

        switch (X.getDataType())
        {
        case TensorDataType::FLOAT32:
            if (use_winograd)
                return executeWinogradConv<float>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
            else
                return executeConv<float>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        case TensorDataType::FLOAT64:
            return executeConv<double>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        case TensorDataType::FLOAT16:
            if (use_winograd)
                return executeWinogradConv<half_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
            else
                return executeConv<half_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
};

#endif