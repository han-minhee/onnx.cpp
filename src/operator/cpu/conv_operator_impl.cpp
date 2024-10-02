#include "operator/operators.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace CPU_OP
{
    bool useWinograd(const std::vector<int64_t> &kernel_shape, const std::vector<int64_t> &strides,
                     const std::vector<int64_t> &dilations, int64_t C, int64_t M, int64_t H, int64_t W_in)
    {
        return false;
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

        if (C < 16 || M < 16 || H < 16 || W_in < 16) // return if it's too small
        {
            return false;
        }

        return true;
    }

    template <typename T>
    OperatorExecuteResult executeConv(
        /// FIXME: change all tensors to use pointers
        // tensors
        const Tensor &X, const Tensor &W, const Tensor *B, Tensor *Y,
        // attributes
        const std::vector<int64_t> &pads, const std::vector<int64_t> &strides, const std::vector<int64_t> &dilations, int64_t group,
        // X shapes
        int64_t N, int64_t C, int64_t H, int64_t W_in,
        // W shapes
        int64_t M, int64_t kH, int64_t kW,
        // Y shapes
        int64_t H_out, int64_t W_out)
    {

        const T *input_data = X.data<T>();
        const T *weight_data = W.data<T>();
        const T *bias_data = B ? B->data<T>() : nullptr;
        T *output_data = Y->data<T>();

        // initialize output tensor to 0
        std::fill(output_data, output_data + (N * M * H_out * W_out), static_cast<T>(0));

        for (int64_t g = 0; g < group; ++g)
        {
            // for N batch
            for (int64_t n = 0; n < N; ++n)
            {
                // for M output channels
                for (int64_t m = 0; m < M / group; ++m)
                {
                    // output index (h,out, w_out)
                    for (int64_t h_out = 0; h_out < H_out; ++h_out)
                    {
                        for (int64_t w_out = 0; w_out < W_out; ++w_out)
                        {
                            int64_t output_idx = n * (M * H_out * W_out) +
                                                 (g * (M / group) + m) * (H_out * W_out) +
                                                 h_out * W_out + w_out;

                            for (int64_t c = 0; c < C / group; ++c)
                            {
                                for (int64_t kh = 0; kh < kH; ++kh)
                                {
                                    for (int64_t kw = 0; kw < kW; ++kw)
                                    {
                                        int64_t h_in = h_out * strides[0] +
                                                       kh * dilations[0] - pads[0];
                                        int64_t w_in = w_out * strides[1] +
                                                       kw * dilations[1] - pads[1];

                                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W_in)
                                        {
                                            int64_t input_idx = n * (C * H * W_in) + (g * (C / group) + c) * (H * W_in) + h_in * W_in + w_in;
                                            int64_t weight_idx = (g * (M / group) + m) * (C / group * kH * kW) + c * (kH * kW) + kh * kW + kw;
                                            output_data[output_idx] += input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }

                            if (bias_data)
                            {
                                output_data[output_idx] += bias_data[g * (M / group) + m];
                            }
                        }
                    }
                }
            }
        }
        return OperatorExecuteResult::SUCCESS;
    }

    template <typename T>
    void winogradTransformKernel(const T kernel[3][3], T transformed_kernel[4][4])
    {
        // G * d * Gt

        const T G[4][3] = {
            {1.0f, 0.0f, 0.0f},
            {0.5f, 0.5f, 0.5f},
            {0.5f, -0.5f, 0.5f},
            {0.0f, 0.0f, 1.0f}};
        const T Gt[3][4] = {
            {1.0f, 0.5f, 0.5f, 0.0f},
            {0.0f, 0.5f, -0.5f, 0.0f},
            {0.0f, 0.5f, 0.5f, 1.0f}};

        T temp[4][3];

        for (int64_t i = 0; i < 4; ++i)
        {
            for (int64_t j = 0; j < 3; ++j)
            {
                temp[i][j] = 0;
                for (int64_t k = 0; k < 3; ++k)
                {
                    temp[i][j] += G[i][k] * kernel[k][j];
                }
            }
        }

        for (int64_t i = 0; i < 4; ++i)
        {
            for (int64_t j = 0; j < 4; ++j)
            {
                transformed_kernel[i][j] = 0;
                for (int64_t k = 0; k < 3; ++k)
                {
                    transformed_kernel[i][j] += temp[i][k] * Gt[k][j];
                }
            }
        }
    }

    template <typename T>
    void winogradTransformInput(const T d[4][4], T transformed_input[4][4])
    {
        // B * d * Bt
        const T Bt[4][4] = {
            {1.0f, 0.0f, -1.0f, 0.0f},
            {0.0f, 1.0f, 1.0f, 0.0f},
            {0.0f, -1.0f, 1.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, -1.0f}};
        const T BtT[4][4] = {
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, -1.0f, 1.0f},
            {-1.0f, 1.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, -1.0f}};

        T temp[4][4];

        for (int64_t i = 0; i < 4; ++i)
        {
            for (int64_t j = 0; j < 4; ++j)
            {
                temp[i][j] = 0;
                for (int64_t k = 0; k < 4; ++k)
                {
                    temp[i][j] += Bt[i][k] * d[k][j];
                }
            }
        }

        for (int64_t i = 0; i < 4; ++i)
        {
            for (int64_t j = 0; j < 4; ++j)
            {
                transformed_input[i][j] = 0;
                for (int64_t k = 0; k < 4; ++k)
                {
                    transformed_input[i][j] += temp[i][k] * BtT[k][j];
                }
            }
        }
    }

    template <typename T>
    void winogradTransformOutput(const T m[4][4], T output_tile[2][2])
    {
        // At * m * A

        const T At[2][4] = {
            {1.0f, 1.0f, 1.0f, 0.0f},
            {0.0f, 1.0f, -1.0f, -1.0f}};
        const T AtT[4][2] = {
            {1.0f, 0.0f},
            {1.0f, 1.0f},
            {1.0f, -1.0f},
            {0.0f, -1.0f}};

        T temp[2][4];

        for (int64_t i = 0; i < 2; ++i)
        {
            for (int64_t j = 0; j < 4; ++j)
            {
                temp[i][j] = 0;
                for (int64_t k = 0; k < 4; ++k)
                {
                    temp[i][j] += AtT[k][i] * m[k][j];
                }
            }
        }

        for (int64_t i = 0; i < 2; ++i)
        {
            for (int64_t j = 0; j < 2; ++j)
            {
                output_tile[i][j] = 0;
                for (int64_t k = 0; k < 4; ++k)
                {
                    output_tile[i][j] += temp[i][k] * At[j][k];
                }
            }
        }
    }

    template <typename T>
    OperatorExecuteResult executeWinogradConv(
        /// FIXME: change all tensors to use pointers
        // tensors
        const Tensor &X, const Tensor &W, const Tensor *B, Tensor *Y,
        // attributes
        const std::vector<int64_t> &pads, const std::vector<int64_t> &strides, const std::vector<int64_t> &dilations, int64_t group,
        // X shapes
        int64_t N, int64_t C, int64_t H, int64_t W_in,
        // W shapes
        int64_t M, int64_t kH, int64_t kW,
        // Y shapes
        int64_t H_out, int64_t W_out)
    {

        const T *input_data = X.data<T>();
        const T *weight_data = W.data<T>();
        const T *bias_data = B ? B->data<T>() : nullptr;

        T *output_data = Y->data<T>();
        std::fill(output_data, output_data + (N * M * H_out * W_out), static_cast<T>(0));

        // the W is (C x M/group x kH x kW)
        // each kernel has size (kH x kW)
        // thus, there are total of C x M/group kernels
        // M * (C / group) number of transformed kernels
        // each transformed kernel has size 4 x 4 = 16
        std::vector<T> transformed_kernels(M * (C / group) * 16);

        for (int64_t g = 0; g < group; ++g)
        {
            // M/group : number of output channels per group
            for (int64_t m = 0; m < M / group; ++m)
            {
                // C/group : number of input channels per group
                for (int64_t c = 0; c < C / group; ++c)
                {

                    // 3x3 kernel (fixed for Winograd)
                    T kernel[3][3];
                    for (int64_t kh = 0; kh < 3; ++kh)
                    {
                        for (int64_t kw = 0; kw < 3; ++kw)
                        {
                            /// XXX: Make it simpler by using strides
                            int64_t weight_idx = (g * (M / group) + m) * (C / group * kH * kW) + c * (kH * kW) + kh * kW + kw;
                            kernel[kh][kw] = weight_data[weight_idx];
                        }
                    }

                    T transformed_kernel[4][4];
                    winogradTransformKernel<T>(kernel, transformed_kernel);

                    int64_t kernel_idx = ((g * (M / group) + m) * (C / group) + c) * 16;
                    for (int64_t i = 0; i < 4; ++i)
                    {
                        for (int64_t j = 0; j < 4; ++j)
                        {
                            transformed_kernels[kernel_idx + i * 4 + j] = transformed_kernel[i][j];
                        }
                    }
                }
            }
        }

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t h_out = 0; h_out < H_out; h_out += 2)
            {
                for (size_t w_out = 0; w_out < W_out; w_out += 2)
                {

                    for (size_t g = 0; g < static_cast<size_t>(group); ++g)
                    {

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
                                        int64_t h_in = static_cast<int64_t>(h_out * strides[0]) + ih - pads[0];
                                        int64_t w_in = static_cast<int64_t>(w_out * strides[1]) + iw - pads[1];

                                        if (h_in >= 0 && h_in < static_cast<int64_t>(H) && w_in >= 0 && w_in < static_cast<int64_t>(W_in))
                                        {
                                            size_t input_idx = n * (C * H * W_in) + (g * (C / group) + c) * (H * W_in) + static_cast<size_t>(h_in) * W_in + static_cast<size_t>(w_in);
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
                                        transformed_kernel[i][j] = transformed_kernels[kernel_idx + i * 4 + j];
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
                                    size_t h = h_out + i;
                                    size_t w = w_out + j;

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
                }
            }
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult ConvOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs, const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        /// FIXME: use .at()
        /// FIXME: Why don't we just use pointers for all tensos as there are cases where the tensor is actually not provided.

        // input tensors
        const Tensor &X = inputs.at(0);
        const Tensor &W = inputs.at(1);
        bool has_bias = inputs.size() == 3;
        const Tensor *B = has_bias ? &inputs.at(2) : nullptr;

        // output tensor
        Tensor *Y = outputs.at(0);

        // get attributes

        /// FIXME: substitute all size_t with int64_t
        // currently, getDims() returns std::vector<size_t>
        // so, we need to convert it to std::vector<int64_t>
        const std::vector<size_t> &X_dims_size_t = X.getDims();
        const std::vector<size_t> &W_dims_size_t = W.getDims();

        std::vector<int64_t> X_dims(X_dims_size_t.begin(), X_dims_size_t.end());
        std::vector<int64_t> W_dims(W_dims_size_t.begin(), W_dims_size_t.end());

        /// FIXME: auto_pad is actually not used now (the code assumes it is NOTSET)
        std::string auto_pad = "NOTSET";
        std::vector<int64_t> dilations = {1, 1};
        /// FIXME: group other than 1 is not tested
        int64_t group = 1;
        std::vector<int64_t> kernel_shape = {W_dims[2], W_dims[3]};
        std::vector<int64_t> pads = {0, 0, 0, 0};
        std::vector<int64_t> strides = {1, 1};

        /// FIXME: move attributes error checking to elsewhere
        for (const auto &[key, value] : attributes)
        {
            if (key == "auto_pad")
                auto_pad = std::get<std::string>(value);

            else if (key == "dilations")
            {
                dilations = std::get<std::vector<int64_t>>(value);
            }
            else if (key == "group")
                group = std::get<int64_t>(value);
            else if (key == "kernel_shape")
            {
                kernel_shape = std::get<std::vector<int64_t>>(value);
            }
            else if (key == "pads")
            {
                pads = std::get<std::vector<int64_t>>(value);
            }
            else if (key == "strides")
            {
                strides = std::get<std::vector<int64_t>>(value);
            }
        }

        int64_t N = X_dims[0];
        int64_t C = X_dims[1];
        int64_t H = X_dims[2];
        int64_t W_in = X_dims[3];

        int64_t M = W_dims[0];
        int64_t kH = kernel_shape[0];
        int64_t kW = kernel_shape[1];

        int64_t H_out = static_cast<int64_t>(
            std::floor((H + pads[0] + pads[2] - ((kH - 1) * dilations[0] + 1)) / static_cast<double>(strides[0])) + 1);
        int64_t W_out = static_cast<int64_t>(
            std::floor((W_in + pads[1] + pads[3] - ((kW - 1) * dilations[1] + 1)) / static_cast<double>(strides[1])) + 1);

        bool use_winograd_flag = useWinograd(kernel_shape, strides, dilations, C, M, H, W_in);

        switch (X.getDataType())
        {
        case TensorDataType::FLOAT32:
            return use_winograd_flag ? executeWinogradConv<float>(
                                           X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out)
                                     : executeConv<float>(
                                           X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);

        case TensorDataType::FLOAT64:
            return use_winograd_flag ? executeWinogradConv<double>(
                                           X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out)
                                     : executeConv<double>(
                                           X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        case TensorDataType::FLOAT16:
            return use_winograd_flag ? executeWinogradConv<half_t>(
                                           X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out)
                                     : executeConv<half_t>(
                                           X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
}
