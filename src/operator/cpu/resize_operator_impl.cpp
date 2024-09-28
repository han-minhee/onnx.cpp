#include "operator/operators.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace CPU_OP
{
    template <typename T>
    float compute_x_original(float x_resized, float scale, const std::string &coordinate_transformation_mode,
                             size_t input_size, size_t output_size)
    {
        if (coordinate_transformation_mode == "half_pixel" || coordinate_transformation_mode == "pytorch_half_pixel")
        {
            return (output_size > 1) ? (x_resized + 0.5f) / scale - 0.5f : 0;
        }
        if (coordinate_transformation_mode == "asymmetric")
        {
            return x_resized / scale;
        }
        if (coordinate_transformation_mode == "align_corners")
        {
            return (output_size == 1) ? 0 : x_resized * (input_size - 1) / (output_size - 1);
        }
        return (x_resized + 0.5f) / scale - 0.5f; // Default behavior
    }

    template <typename T>
    size_t compute_x_input(float x_original, const std::string &nearest_mode, size_t input_size)
    {
        float x_nearest = (nearest_mode == "round_prefer_floor")  ? std::floor(x_original + 0.5f)
                          : (nearest_mode == "round_prefer_ceil") ? std::ceil(x_original - 0.5f)
                          : (nearest_mode == "floor")             ? std::floor(x_original)
                                                                  : std::ceil(x_original);

        return static_cast<size_t>(std::clamp(x_nearest, 0.0f, static_cast<float>(input_size - 1)));
    }

    template <typename T>
    OperatorExecuteResult executeResize(const Tensor &input_tensor, Tensor *output_tensor,
                                        const std::unordered_map<std::string, Node::AttributeValue> &attributes,
                                        const std::vector<float> &scales, const std::vector<int64_t> &sizes,
                                        const std::string &mode, const std::string &coordinate_transformation_mode,
                                        const std::string &nearest_mode,
                                        bool has_scales, bool has_sizes)
    {
        const T *input_data = input_tensor.data<T>();
        auto input_shape = input_tensor.getDims();
        auto input_strides = input_tensor.getStrides();

        std::vector<size_t> output_shape(input_shape.size());
        std::vector<float> adjusted_scales(scales);

        if (has_sizes)
        {
            output_shape.assign(sizes.begin(), sizes.end());
            for (size_t i = 0; i < output_shape.size(); ++i)
            {
                adjusted_scales[i] = static_cast<float>(output_shape[i]) / input_shape[i];
            }
        }
        else if (has_scales)
        {
            for (size_t i = 0; i < input_shape.size(); ++i)
            {
                output_shape[i] = static_cast<size_t>(std::floor(input_shape[i] * scales[i]));
            }
        }
        else
        {
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        }

        size_t num_output_elements = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
        output_tensor->reshape(output_shape);
        output_tensor->setDataType(input_tensor.getDataType());

        if (!output_tensor->data<T>() || output_tensor->getNumElements() != num_output_elements)
        {
            output_tensor->allocateBuffer(input_tensor.getDataType(), num_output_elements);
        }

        T *output_data = output_tensor->data<T>();
        auto output_strides = output_tensor->getStrides();

        std::vector<size_t> indices(output_shape.size(), 0);

        while (true)
        {
            size_t output_index = std::inner_product(indices.begin(), indices.end(), output_strides.begin(), size_t(0));

            std::vector<size_t> x_input(input_shape.size());
            for (size_t i = 0; i < input_shape.size(); ++i)
            {
                float x_orig = compute_x_original<T>(static_cast<float>(indices[i]), adjusted_scales[i],
                                                     coordinate_transformation_mode, input_shape[i], output_shape[i]);
                x_input[i] = compute_x_input<T>(x_orig, nearest_mode, input_shape[i]);
            }

            size_t input_index = std::inner_product(x_input.begin(), x_input.end(), input_strides.begin(), size_t(0));
            output_data[output_index] = input_data[input_index];

            size_t dim = output_shape.size();
            for (int i = static_cast<int>(dim) - 1; i >= 0; --i)
            {
                if (++indices[i] < output_shape[i])
                    break;
                if (i == 0)
                    return OperatorExecuteResult::SUCCESS;
                indices[i] = 0;
            }
        }
        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult ResizeOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                      const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        const Tensor &input_tensor = inputs[0];
        Tensor *output_tensor = outputs[0];

        std::string mode = attributes.count("mode") ? std::get<std::string>(attributes.at("mode")) : "nearest";
        std::string coordinate_transformation_mode = attributes.count("coordinate_transformation_mode")
                                                         ? std::get<std::string>(attributes.at("coordinate_transformation_mode"))
                                                         : "half_pixel";
        std::string nearest_mode = attributes.count("nearest_mode") ? std::get<std::string>(attributes.at("nearest_mode"))
                                                                    : "round_prefer_floor";

        bool has_scales = inputs.size() >= 3;
        bool has_sizes = inputs.size() == 4;
        std::vector<float> scales;
        std::vector<int64_t> sizes;

        if (has_scales)
        {
            const Tensor &scales_tensor = inputs[2];
            if (scales_tensor.getDataType() != TensorDataType::FLOAT32)
                return OperatorExecuteResult::DATA_TYPE_ERROR;

            scales.assign(scales_tensor.data<float>(), scales_tensor.data<float>() + scales_tensor.getNumElements());
        }

        if (has_sizes)
        {
            const Tensor &sizes_tensor = inputs[3];
            if (sizes_tensor.getDataType() != TensorDataType::INT64)
                return OperatorExecuteResult::DATA_TYPE_ERROR;

            sizes.assign(sizes_tensor.data<int64_t>(), sizes_tensor.data<int64_t>() + sizes_tensor.getNumElements());
        }

        if (has_scales == has_sizes)
            return OperatorExecuteResult::ATTRIBUTE_ERROR;

        switch (input_tensor.getDataType())
        {
        case TensorDataType::FLOAT32:
            return executeResize<float>(input_tensor, output_tensor, attributes, scales, sizes, mode, coordinate_transformation_mode, nearest_mode, has_scales, has_sizes);
        case TensorDataType::INT32:
            return executeResize<int32_t>(input_tensor, output_tensor, attributes, scales, sizes, mode, coordinate_transformation_mode, nearest_mode, has_scales, has_sizes);
        case TensorDataType::INT64:
            return executeResize<int64_t>(input_tensor, output_tensor, attributes, scales, sizes, mode, coordinate_transformation_mode, nearest_mode, has_scales, has_sizes);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
}
