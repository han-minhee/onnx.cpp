#include "operator/operators.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

std::vector<std::vector<size_t>> ConvOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                 const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    const Tensor &X = inputs.at(0);
    const Tensor &W = inputs.at(1);

    // Get attributes with defaults
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    int64_t group = 1;

    for (const auto &[key, value] : attributes)
    {
        if (key == "pads")
            pads = std::get<std::vector<int64_t>>(value);
        else if (key == "strides")
            strides = std::get<std::vector<int64_t>>(value);
        else if (key == "dilations")
            dilations = std::get<std::vector<int64_t>>(value);
        else if (key == "group")
            group = std::get<int64_t>(value);
    }
    const std::vector<size_t> &X_dims = X.getDims(); // (N, C, H, W)
    const std::vector<size_t> &W_dims = W.getDims(); // (M, C/group, kH, kW)

    // Check dimensions
    if (X_dims.size() != 4 || W_dims.size() != 4)
    {
        return {};
    }

    size_t N = X_dims[0];
    size_t C = X_dims[1];
    size_t H = X_dims[2];
    size_t W_in = X_dims[3];
    size_t M = W_dims[0];
    size_t kH = W_dims[2];
    size_t kW = W_dims[3];

    // Validate kernel shape
    if (kH > H + pads[0] + pads[2] || kW > W_in + pads[1] + pads[3])
    {
        return {}; // Shape mismatch
    }

    // Calculate output dimensions
    size_t H_out = static_cast<size_t>(std::floor((H + pads[0] + pads[2] - kH * dilations[0] + strides[0]) / strides[0]));
    size_t W_out = static_cast<size_t>(std::floor((W_in + pads[1] + pads[3] - kW * dilations[1] + strides[1]) / strides[1]));

    return {{N, M, H_out, W_out}};
}

std::vector<TensorDataType> ConvOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                               const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {TensorDataType::UNDEFINED};
    }

    return {inputs.at(0).getDataType()};
}

template <typename T>
OperatorExecuteResult executeConvolution(const Tensor &X, const Tensor &W, const Tensor *B, Tensor *Y,
                                         const std::vector<int64_t> &pads, const std::vector<int64_t> &strides,
                                         const std::vector<int64_t> &dilations, int64_t group, size_t N, size_t C,
                                         size_t H, size_t W_in, size_t M, size_t kH, size_t kW, size_t H_out, size_t W_out)
{
    const T *input_data = X.data<T>();
    const T *weight_data = W.data<T>();
    const T *bias_data = B ? B->data<T>() : nullptr;

    if (!input_data || !weight_data || (B && !bias_data))
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    T *output_data = new (std::nothrow) T[N * M * H_out * W_out]();
    if (!output_data)
    {
        return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
    }

    // Perform convolution
    // Check `group` size in executeConvolution loop
    for (size_t g = 0; g < group; ++g)
    {
        for (size_t n = 0; n < N; ++n) // batch
        {
            for (size_t m = 0; m < M / group; ++m) // output channels per group
            {
                for (size_t h_out = 0; h_out < H_out; ++h_out)
                {
                    for (size_t w_out = 0; w_out < W_out; ++w_out)
                    {
                        T sum = 0;

                        // Apply kernel over input channels
                        for (size_t c = 0; c < C / group; ++c)
                        {
                            for (size_t kh = 0; kh < kH; ++kh)
                            {
                                for (size_t kw = 0; kw < kW; ++kw)
                                {
                                    int h_in = static_cast<int>(h_out * strides[0] + kh * dilations[0] - pads[0]);
                                    int w_in = static_cast<int>(w_out * strides[1] + kw * dilations[1] - pads[1]);

                                    if (h_in >= 0 && h_in < static_cast<int>(H) && w_in >= 0 && w_in < static_cast<int>(W_in))
                                    {
                                        size_t input_idx = n * (C * H * W_in) + (g * C / group + c) * (H * W_in) + h_in * W_in + w_in;
                                        size_t weight_idx = (g * M / group + m) * (C / group * kH * kW) + c * (kH * kW) + kh * kW + kw;
                                        sum += input_data[input_idx] * weight_data[weight_idx];
                                    }
                                }
                            }
                        }

                        if (bias_data)
                        {
                            sum += bias_data[g * M / group + m];
                        }

                        size_t output_idx = n * (M * H_out * W_out) + (g * M / group + m) * (H_out * W_out) + h_out * W_out + w_out;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
    }

    Y->setDataType(X.getDataType());
    Y->setDataPointer<T>(output_data, {N, M, H_out, W_out});

    return OperatorExecuteResult::SUCCESS;
}

OperatorExecuteResult ConvOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                            const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() < 2)
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

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
        return OperatorExecuteResult::SHAPE_MISMATCH_ERROR; // Conv operator supports 4D tensors
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

    size_t H_out = static_cast<size_t>(std::floor((H + pads[0] + pads[2] - (kH - 1) * dilations[0] - 1) / strides[0] + 1));
    size_t W_out = static_cast<size_t>(std::floor((W_in + pads[1] + pads[3] - (kW - 1) * dilations[1] - 1) / strides[1] + 1));

    Tensor *Y = outputs[0];

    switch (X.getDataType())
    {
    case TensorDataType::FLOAT32:
        return executeConvolution<float>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
    case TensorDataType::FLOAT64:
        return executeConvolution<double>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
    case TensorDataType::INT32:
        return executeConvolution<int32_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
    case TensorDataType::INT64:
        return executeConvolution<int64_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
    case TensorDataType::INT8:
        return executeConvolution<int8_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
    case TensorDataType::UINT8:
        return executeConvolution<uint8_t>(X, W, B, Y, pads, strides, dilations, group, N, C, H, W_in, M, kH, kW, H_out, W_out);
    default:
        return OperatorExecuteResult::DATA_TYPE_ERROR;
    }
}
