#include "operator/operators.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

std::vector<std::vector<size_t>> MaxPoolOperator::inferOutputShapes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    const Tensor &X = inputs.at(0);
    const std::vector<size_t> &X_dims = X.getDims();

    if (X_dims.size() < 4)
    {
        return {};
    }

    size_t spatial_rank = X_dims.size() - 2;

    // Initialize vectors with default values
    std::vector<int64_t> kernel_shape(spatial_rank, 1);
    std::vector<int64_t> pads(spatial_rank * 2, 0);
    std::vector<int64_t> strides(spatial_rank, 1);
    std::vector<int64_t> dilations(spatial_rank, 1);
    std::string auto_pad = "NOTSET";
    int64_t ceil_mode = 0;

    // Extract attribute values and resize vectors accordingly
    for (const auto &[key, value] : attributes)
    {
        if (key == "kernel_shape")
        {
            kernel_shape = std::get<std::vector<int64_t>>(value);
            if (kernel_shape.size() != spatial_rank)
            {
                kernel_shape.resize(spatial_rank, 1); // Ensure correct size
            }
        }
        else if (key == "pads")
        {
            pads = std::get<std::vector<int64_t>>(value);
            if (pads.size() != spatial_rank * 2)
            {
                pads.resize(spatial_rank * 2, 0); // Ensure correct size
            }
        }
        else if (key == "strides")
        {
            strides = std::get<std::vector<int64_t>>(value);
            if (strides.size() != spatial_rank)
            {
                strides.resize(spatial_rank, 1); // Ensure correct size
            }
        }
        else if (key == "dilations")
        {
            dilations = std::get<std::vector<int64_t>>(value);
            if (dilations.size() != spatial_rank)
            {
                dilations.resize(spatial_rank, 1); // Ensure correct size
            }
        }
        else if (key == "auto_pad")
        {
            auto_pad = std::get<std::string>(value);
        }
        else if (key == "ceil_mode")
        {
            ceil_mode = std::get<int64_t>(value);
        }
    }

    std::vector<size_t> output_dims = {X_dims[0], X_dims[1]};

    for (size_t i = 0; i < spatial_rank; ++i)
    {
        int64_t input_size = X_dims[i + 2];
        int64_t kernel = kernel_shape[i];
        int64_t pad_total = pads[i] + pads[i + spatial_rank];
        int64_t dilation = dilations[i];
        int64_t stride = strides[i];

        int64_t output_size = 0;

        if (auto_pad == "VALID")
        {
            output_size = (input_size - dilation * (kernel - 1) - 1) / stride + 1;
        }
        else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER")
        {
            output_size = static_cast<int64_t>(std::ceil(input_size / static_cast<double>(stride)));
        }
        else
        {
            int64_t numerator = input_size + pad_total - dilation * (kernel - 1) - 1;
            if (ceil_mode)
            {
                output_size = static_cast<int64_t>(std::ceil(numerator / static_cast<double>(stride))) + 1;
            }
            else
            {
                output_size = static_cast<int64_t>(std::floor(numerator / static_cast<double>(stride))) + 1;
            }
        }

        output_dims.push_back(static_cast<size_t>(std::max<int64_t>(output_size, 1)));
    }

    return {output_dims};
}

std::vector<TensorDataType> MaxPoolOperator::inferOutputDataTypes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {TensorDataType::UNDEFINED};
    }

    return {inputs.at(0).getDataType()};
}
template <typename T>
OperatorExecuteResult executeMaxPool(const Tensor &X, Tensor *Y,
                                     const std::vector<int64_t> &kernel_shape,
                                     const std::vector<int64_t> &pads,
                                     const std::vector<int64_t> &strides,
                                     const std::vector<int64_t> &dilations,
                                     size_t N, size_t C,
                                     const std::vector<size_t> &input_spatial_shape,
                                     const std::vector<size_t> &output_spatial_shape,
                                     size_t spatial_rank)
{
    const T *input_data = X.data<T>();
    if (!input_data)
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    size_t output_size = Y->getNumElements();
    T *output_data = new (std::nothrow) T[output_size];
    if (!output_data)
    {
        return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
    }

    for (size_t n = 0; n < N; ++n)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t out_h = 0; out_h < output_spatial_shape[0]; ++out_h)
            {
                for (size_t out_w = 0; out_w < output_spatial_shape[1]; ++out_w)
                {
                    T max_val = std::numeric_limits<T>::lowest();

                    for (size_t kh = 0; kh < kernel_shape[0]; ++kh)
                    {
                        for (size_t kw = 0; kw < kernel_shape[1]; ++kw)
                        {
                            int h_in = static_cast<int>(out_h * strides[0] + kh * dilations[0] - pads[0]);
                            int w_in = static_cast<int>(out_w * strides[1] + kw * dilations[1] - pads[1]);

                            if (h_in >= 0 && h_in < static_cast<int>(input_spatial_shape[0]) &&
                                w_in >= 0 && w_in < static_cast<int>(input_spatial_shape[1]))
                            {
                                size_t input_idx = n * C * input_spatial_shape[0] * input_spatial_shape[1] +
                                                   c * input_spatial_shape[0] * input_spatial_shape[1] +
                                                   h_in * input_spatial_shape[1] + w_in;
                                max_val = std::max(max_val, input_data[input_idx]);
                            }
                        }
                    }
                    size_t output_idx = n * C * output_spatial_shape[0] * output_spatial_shape[1] +
                                        c * output_spatial_shape[0] * output_spatial_shape[1] +
                                        out_h * output_spatial_shape[1] + out_w;
                    output_data[output_idx] = max_val;
                }
            }
        }
    }

    Y->setDataType(X.getDataType());
    Y->setDataPointer<T>(output_data, {N, C, output_spatial_shape[0], output_spatial_shape[1]});

    return OperatorExecuteResult::SUCCESS;
}

OperatorExecuteResult MaxPoolOperator::execute(
    const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() < 1 || outputs.empty() || outputs[0] == nullptr)
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    const Tensor &X = inputs.at(0);
    Tensor *Y = outputs[0];

    const std::vector<size_t> &X_dims = X.getDims();
    size_t N = X_dims[0];
    size_t C = X_dims[1];
    size_t spatial_rank = X_dims.size() - 2;

    std::string auto_pad = "NOTSET";
    int64_t ceil_mode = 0;
    std::vector<int64_t> dilations;
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    int64_t storage_order = 0;
    std::vector<int64_t> strides;

    for (const auto &[key, value] : attributes)
    {
        if (key == "auto_pad")
            auto_pad = std::get<std::string>(value);
        else if (key == "ceil_mode")
            ceil_mode = std::get<int64_t>(value);
        else if (key == "dilations")
            dilations = std::get<std::vector<int64_t>>(value);
        else if (key == "kernel_shape")
            kernel_shape = std::get<std::vector<int64_t>>(value);
        else if (key == "pads")
            pads = std::get<std::vector<int64_t>>(value);
        else if (key == "storage_order")
            storage_order = std::get<int64_t>(value);
        else if (key == "strides")
            strides = std::get<std::vector<int64_t>>(value);
    }

    // Add this block to ensure the vectors have correct sizes
    kernel_shape.resize(spatial_rank, 1);
    pads.resize(spatial_rank * 2, 0);
    strides.resize(spatial_rank, 1);
    dilations.resize(spatial_rank, 1);

    std::vector<size_t> input_spatial_shape(X_dims.begin() + 2, X_dims.end());
    std::vector<std::vector<size_t>> output_shape = inferOutputShapes(inputs, attributes);
    if (output_shape.empty())
    {
        return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
    }

    std::vector<size_t> output_spatial_shape = output_shape[0];
    output_spatial_shape.erase(output_spatial_shape.begin(), output_spatial_shape.begin() + 2);

    switch (X.getDataType())
    {
    case TensorDataType::FLOAT32:
        return executeMaxPool<float>(X, Y, kernel_shape, pads, strides, dilations, N, C, input_spatial_shape, output_spatial_shape, spatial_rank);
    case TensorDataType::FLOAT64:
        return executeMaxPool<double>(X, Y, kernel_shape, pads, strides, dilations, N, C, input_spatial_shape, output_spatial_shape, spatial_rank);
    case TensorDataType::INT32:
        return executeMaxPool<int32_t>(X, Y, kernel_shape, pads, strides, dilations, N, C, input_spatial_shape, output_spatial_shape, spatial_rank);
    case TensorDataType::INT64:
        return executeMaxPool<int64_t>(X, Y, kernel_shape, pads, strides, dilations, N, C, input_spatial_shape, output_spatial_shape, spatial_rank);
    case TensorDataType::INT8:
        return executeMaxPool<int8_t>(X, Y, kernel_shape, pads, strides, dilations, N, C, input_spatial_shape, output_spatial_shape, spatial_rank);
    case TensorDataType::UINT8:
        return executeMaxPool<uint8_t>(X, Y, kernel_shape, pads, strides, dilations, N, C, input_spatial_shape, output_spatial_shape, spatial_rank);
    default:
        return OperatorExecuteResult::DATA_TYPE_ERROR;
    }
}
