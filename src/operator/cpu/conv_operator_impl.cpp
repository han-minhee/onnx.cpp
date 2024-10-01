#include "operator/operators.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace CPU_OP
{
    bool useWinograd(const std::vector<size_t> &kernel_shape, const std::vector<size_t> &strides,
                     const std::vector<size_t> &dilations, size_t C, size_t M, size_t H, size_t W_in)
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
    OperatorExecuteResult executeConv(const Tensor &X, const Tensor &W, const Tensor *B, Tensor *Y,
                                      const std::vector<int64_t> &pads, const std::vector<size_t> &strides,
                                      const std::vector<size_t> &dilations, int64_t group, size_t N, size_t C,
                                      size_t H, size_t W_in, size_t M, size_t kH, size_t kW, size_t H_out, size_t W_out)
    {

        const T *input_data = X.data<T>();
        const T *weight_data = W.data<T>();
        const T *bias_data = B ? B->data<T>() : nullptr;

        T *output_data = Y->data<T>();
        std::fill(output_data, output_data + (N * M * H_out * W_out), static_cast<T>(0));

        for (size_t g = 0; g < static_cast<size_t>(group); ++g)
        {
            for (size_t n = 0; n < N; ++n)
            {
                for (size_t m = 0; m < M / group; ++m)
                {
                    for (size_t h_out = 0; h_out < H_out; ++h_out)
                    {
                        for (size_t w_out = 0; w_out < W_out; ++w_out)
                        {
                            T sum = 0;

                            for (size_t c = 0; c < C / group; ++c)
                            {
                                for (size_t kh = 0; kh < kH; ++kh)
                                {
                                    for (size_t kw = 0; kw < kW; ++kw)
                                    {
                                        int64_t h_in = static_cast<int64_t>(h_out) * static_cast<int64_t>(strides[0]) +
                                                       static_cast<int64_t>(kh) * static_cast<int64_t>(dilations[0]) - pads[0];
                                        int64_t w_in = static_cast<int64_t>(w_out) * static_cast<int64_t>(strides[1]) +
                                                       static_cast<int64_t>(kw) * static_cast<int64_t>(dilations[1]) - pads[1];

                                        if (h_in >= 0 && h_in < static_cast<int64_t>(H) && w_in >= 0 && w_in < static_cast<int64_t>(W_in))
                                        {
                                            size_t input_idx = n * (C * H * W_in) + (g * (C / group) + c) * (H * W_in) + static_cast<size_t>(h_in) * W_in + static_cast<size_t>(w_in);
                                            size_t weight_idx = (g * (M / group) + m) * (C / group * kH * kW) + c * (kH * kW) + kh * kW + kw;
                                            sum += input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }

                            if (bias_data)
                            {
                                sum += bias_data[g * (M / group) + m];
                            }

                            size_t output_idx = n * (M * H_out * W_out) +
                                                (g * (M / group) + m) * (H_out * W_out) +
                                                h_out * W_out + w_out;
                            output_data[output_idx] = sum;
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
    void winogradTransformInput(const T d[4][4], T transformed_input[4][4])
    {

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

        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                temp[i][j] = 0;
                for (size_t k = 0; k < 4; ++k)
                {
                    temp[i][j] += Bt[i][k] * d[k][j];
                }
            }
        }

        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                transformed_input[i][j] = 0;
                for (size_t k = 0; k < 4; ++k)
                {
                    transformed_input[i][j] += temp[i][k] * BtT[k][j];
                }
            }
        }
    }

    template <typename T>
    void winogradTransformOutput(const T m[4][4], T output_tile[2][2])
    {

        const T At[2][4] = {
            {1.0f, 1.0f, 1.0f, 0.0f},
            {0.0f, 1.0f, -1.0f, -1.0f}};
        const T AtT[4][2] = {
            {1.0f, 0.0f},
            {1.0f, 1.0f},
            {1.0f, -1.0f},
            {0.0f, -1.0f}};

        T temp[2][4];

        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                temp[i][j] = 0;
                for (size_t k = 0; k < 4; ++k)
                {
                    temp[i][j] += AtT[k][i] * m[k][j];
                }
            }
        }

        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                output_tile[i][j] = 0;
                for (size_t k = 0; k < 4; ++k)
                {
                    output_tile[i][j] += temp[i][k] * At[j][k];
                }
            }
        }
    }

    template <typename T>
    OperatorExecuteResult executeWinogradConv(const Tensor &X, const Tensor &W, const Tensor *B, Tensor *Y,
                                              const std::vector<int64_t> &pads, const std::vector<size_t> &strides,
                                              const std::vector<size_t> &dilations, int64_t group, size_t N, size_t C,
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
        std::fill(output_data, output_data + (N * M * H_out * W_out), static_cast<T>(0));

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
        const Tensor &X = inputs.at(0);
        const Tensor &W = inputs.at(1);

        bool has_bias = inputs.size() == 3;
        const Tensor *B = has_bias ? &inputs.at(2) : nullptr;

        Tensor *Y = outputs[0];

        const std::vector<size_t> &X_dims = X.getDims();
        const std::vector<size_t> &W_dims = W.getDims();

        std::string auto_pad = "NOTSET";
        std::vector<size_t> dilations = {1, 1};
        int64_t group = 1;
        std::vector<size_t> kernel_shape = {static_cast<size_t>(W_dims[2]), static_cast<size_t>(W_dims[3])};
        std::vector<int64_t> pads = {0, 0, 0, 0};
        std::vector<size_t> strides = {1, 1};

        for (const auto &[key, value] : attributes)
        {
            if (key == "auto_pad")
                auto_pad = std::get<std::string>(value);
            else if (key == "dilations")
            {

                auto temp_dilations = std::get<std::vector<int64_t>>(value);
                dilations.clear();
                for (const auto &d : temp_dilations)
                {
                    if (d < 0)
                        return OperatorExecuteResult::ATTRIBUTE_ERROR;
                    dilations.push_back(static_cast<size_t>(d));
                }
            }
            else if (key == "group")
                group = std::get<int64_t>(value);
            else if (key == "kernel_shape")
            {
                auto temp_kernel = std::get<std::vector<int64_t>>(value);
                kernel_shape.clear();
                for (const auto &k : temp_kernel)
                {
                    if (k <= 0)
                        return OperatorExecuteResult::ATTRIBUTE_ERROR;
                    kernel_shape.push_back(static_cast<size_t>(k));
                }
            }
            else if (key == "pads")
            {
                pads = std::get<std::vector<int64_t>>(value);
                if (pads.size() != 4)
                    return OperatorExecuteResult::ATTRIBUTE_ERROR;
            }
            else if (key == "strides")
            {
                auto temp_strides = std::get<std::vector<int64_t>>(value);
                strides.clear();
                for (const auto &s : temp_strides)
                {
                    if (s <= 0)
                        return OperatorExecuteResult::ATTRIBUTE_ERROR;
                    strides.push_back(static_cast<size_t>(s));
                }
            }
        }

        size_t N = X_dims[0];
        size_t C = X_dims[1];
        size_t H = X_dims[2];
        size_t W_in = X_dims[3];
        size_t M = W_dims[0];
        size_t kH = kernel_shape[0];
        size_t kW = kernel_shape[1];

        if (group <= 0 || C % group != 0 || M % group != 0)
        {
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        }

        int64_t H_out_f = static_cast<int64_t>(std::floor((static_cast<double>(H) + pads[0] + pads[2] - (static_cast<int64_t>(kH) - 1) * static_cast<int64_t>(dilations[0]) - 1) / static_cast<double>(strides[0]) + 1));
        int64_t W_out_f = static_cast<int64_t>(std::floor((static_cast<double>(W_in) + pads[1] + pads[3] - (static_cast<int64_t>(kW) - 1) * static_cast<int64_t>(dilations[1]) - 1) / static_cast<double>(strides[1]) + 1));

        if (H_out_f <= 0 || W_out_f <= 0)
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        size_t H_out = static_cast<size_t>(H_out_f);
        size_t W_out = static_cast<size_t>(W_out_f);

        std::vector<size_t> kernel_shape_size = kernel_shape;
        std::vector<size_t> strides_size = strides;
        std::vector<size_t> dilations_size = dilations;

        bool use_winograd_flag = useWinograd(kernel_shape_size, strides_size, dilations_size, C, M, H, W_in);

        switch (X.getDataType())
        {
        case TensorDataType::FLOAT32:
            return use_winograd_flag ? executeWinogradConv<float>(
                                           X, W, B, Y, pads, strides_size, dilations_size, group, N, C, H, W_in, M, kH, kW, H_out, W_out)
                                     : executeConv<float>(
                                           X, W, B, Y, pads, strides_size, dilations_size, group, N, C, H, W_in, M, kH, kW, H_out, W_out);

        case TensorDataType::FLOAT64:
            return use_winograd_flag ? executeWinogradConv<double>(
                                           X, W, B, Y, pads, strides_size, dilations_size, group, N, C, H, W_in, M, kH, kW, H_out, W_out)
                                     : executeConv<double>(
                                           X, W, B, Y, pads, strides_size, dilations_size, group, N, C, H, W_in, M, kH, kW, H_out, W_out);

        case TensorDataType::INT32:
            return executeConv<int32_t>(
                X, W, B, Y, pads, strides_size, dilations_size, group, N, C, H, W_in, M, kH, kW, H_out, W_out);

        case TensorDataType::INT64:
            return executeConv<int64_t>(
                X, W, B, Y, pads, strides_size, dilations_size, group, N, C, H, W_in, M, kH, kW, H_out, W_out);

        case TensorDataType::INT8:
            return executeConv<int8_t>(
                X, W, B, Y, pads, strides_size, dilations_size, group, N, C, H, W_in, M, kH, kW, H_out, W_out);

        case TensorDataType::UINT8:
            return executeConv<uint8_t>(
                X, W, B, Y, pads, strides_size, dilations_size, group, N, C, H, W_in, M, kH, kW, H_out, W_out);

        case TensorDataType::FLOAT16:
            return use_winograd_flag ? executeWinogradConv<half_t>(
                                           X, W, B, Y, pads, strides_size, dilations_size, group, N, C, H, W_in, M, kH, kW, H_out, W_out)
                                     : executeConv<half_t>(
                                           X, W, B, Y, pads, strides_size, dilations_size, group, N, C, H, W_in, M, kH, kW, H_out, W_out);

        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }

}
