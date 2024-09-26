#include "operator/operators.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace CPU_OP
{
    template <typename T>
    float compute_x_original(float x_resized, float scale, const std::string &coordinate_transformation_mode,
                             size_t input_size, size_t output_size)
    {
        if (coordinate_transformation_mode == "half_pixel")
        {
            return (x_resized + 0.5f) / scale - 0.5f;
        }
        else if (coordinate_transformation_mode == "asymmetric")
        {
            return x_resized / scale;
        }
        else if (coordinate_transformation_mode == "align_corners")
        {
            if (output_size == 1)
                return 0; // Prevent division by zero
            return x_resized * (input_size - 1) / (output_size - 1);
        }
        else if (coordinate_transformation_mode == "pytorch_half_pixel")
        {
            if (output_size > 1)
                return (x_resized + 0.5f) / scale - 0.5f;
            else
                return 0;
        }
        else
        {
            // Default to 'half_pixel'
            return (x_resized + 0.5f) / scale - 0.5f;
        }
    }

    template <typename T>
    size_t compute_x_input(float x_original, const std::string &nearest_mode, size_t input_size)
    {
        float x_nearest;
        if (nearest_mode == "round_prefer_floor")
        {
            x_nearest = std::floor(x_original + 0.5f);
        }
        else if (nearest_mode == "round_prefer_ceil")
        {
            x_nearest = std::ceil(x_original - 0.5f);
        }
        else if (nearest_mode == "floor")
        {
            x_nearest = std::floor(x_original);
        }
        else if (nearest_mode == "ceil")
        {
            x_nearest = std::ceil(x_original);
        }
        else
        {
            // Default to 'round_prefer_floor'
            x_nearest = std::floor(x_original + 0.5f);
        }

        // Clip x_input[i] to be within [0, input_size - 1]
        if (x_nearest < 0)
            x_nearest = 0;
        if (x_nearest > input_size - 1)
            x_nearest = input_size - 1;

        return static_cast<size_t>(x_nearest);
    }

    std::vector<size_t> calcStrides(const std::vector<size_t> &dims)
    {
        std::vector<size_t> stride(dims.size(), 1);
        for (int i = dims.size() - 2; i >= 0; --i)
        {
            stride[i] = stride[i + 1] * dims[i + 1];
        }
        return stride;
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
        std::vector<size_t> input_shape = input_tensor.getDims();
        std::vector<size_t> input_strides = input_tensor.getStrides();

        std::vector<size_t> output_shape;
        std::vector<float> adjusted_scales = scales;

        if (has_sizes)
        {
            output_shape = std::vector<size_t>(sizes.begin(), sizes.end());
            adjusted_scales.resize(output_shape.size());
            for (size_t i = 0; i < output_shape.size(); ++i)
            {
                adjusted_scales[i] = static_cast<float>(output_shape[i]) / input_shape[i];
            }
        }
        else if (has_scales)
        {
            output_shape.resize(input_shape.size());
            for (size_t i = 0; i < input_shape.size(); ++i)
            {
                output_shape[i] = static_cast<size_t>(std::floor(input_shape[i] * scales[i]));
            }
        }
        else
        {
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        }

        size_t num_output_elements = 1;
        for (size_t i = 0; i < output_shape.size(); ++i)
        {
            num_output_elements *= output_shape[i];
        }

        // Ensure the output tensor is properly allocated
        output_tensor->reshape(output_shape);
        output_tensor->setDataType(input_tensor.getDataType());

        if (!output_tensor->data<T>() || output_tensor->getNumElements() != num_output_elements)
        {
            output_tensor->allocateBuffer(input_tensor.getDataType(), num_output_elements);
        }

        T *output_data = output_tensor->data<T>();

        // Compute output strides
        std::vector<size_t> output_strides = calcStrides(output_shape);

        // Main loop to compute resized tensor
        std::vector<size_t> indices(output_shape.size(), 0);
        while (true)
        {
            // Compute output linear index
            size_t output_index = 0;
            for (size_t i = 0; i < output_shape.size(); ++i)
            {
                output_index += indices[i] * output_strides[i];
            }

            // Compute corresponding input indices
            std::vector<float> x_original(input_shape.size());
            std::vector<size_t> x_input(input_shape.size());

            for (size_t i = 0; i < input_shape.size(); ++i)
            {
                float x_resized = static_cast<float>(indices[i]);
                float scale = adjusted_scales[i];
                float x_orig = compute_x_original<T>(x_resized, scale, coordinate_transformation_mode, input_shape[i], output_shape[i]);

                size_t x_in = compute_x_input<T>(x_orig, nearest_mode, input_shape[i]);

                x_input[i] = x_in;
            }

            // Compute input linear index
            size_t input_index = 0;
            for (size_t i = 0; i < input_shape.size(); ++i)
            {
                input_index += x_input[i] * input_strides[i];
            }

            // Copy value
            output_data[output_index] = input_data[input_index];

            // Increment indices
            size_t dim = output_shape.size();
            for (int i = dim - 1; i >= 0; --i)
            {
                indices[i]++;
                if (indices[i] < output_shape[i])
                {
                    break;
                }
                else if (i > 0)
                {
                    indices[i] = 0;
                }
                else
                {
                    return OperatorExecuteResult::SUCCESS;
                }
            }
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult ResizeOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                      const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        // Validate the number of inputs
        if (inputs.size() < 1 || inputs.size() > 4)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        // Validate the output tensor
        if (outputs.empty() || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const Tensor &input_tensor = inputs[0];
        Tensor *output_tensor = outputs[0];

        // Extract required attributes
        std::string mode = "nearest";
        if (attributes.find("mode") != attributes.end())
        {
            mode = std::get<std::string>(attributes.at("mode"));
        }

        std::string coordinate_transformation_mode = "half_pixel";
        if (attributes.find("coordinate_transformation_mode") != attributes.end())
        {
            coordinate_transformation_mode = std::get<std::string>(attributes.at("coordinate_transformation_mode"));
        }

        std::string nearest_mode = "round_prefer_floor";
        if (attributes.find("nearest_mode") != attributes.end())
        {
            nearest_mode = std::get<std::string>(attributes.at("nearest_mode"));
        }

        // Extract optional attributes
        bool antialias = false;
        if (attributes.find("antialias") != attributes.end())
        {
            antialias = std::get<int64_t>(attributes.at("antialias")) == 1;
        }

        float cubic_coeff_a = -0.75f;
        if (attributes.find("cubic_coeff_a") != attributes.end())
        {
            cubic_coeff_a = std::get<float>(attributes.at("cubic_coeff_a"));
        }

        // Validate the scaling factors or sizes input
        std::vector<float> scales;
        std::vector<int64_t> sizes;
        bool has_scales = false;
        bool has_sizes = false;

        // FIXME: roi tensor input is not handled

        if (inputs.size() >= 3)
        {
            // Scale tensor input
            const Tensor &scales_tensor = inputs[2];
            if (scales_tensor.getDataType() != TensorDataType::FLOAT32)
            {
                return OperatorExecuteResult::DATA_TYPE_ERROR;
            }

            scales = std::vector<float>(scales_tensor.data<float>(), scales_tensor.data<float>() + scales_tensor.getNumElements());
            has_scales = true;
        }

        if (inputs.size() == 4)
        {
            // Sizes tensor input
            const Tensor &sizes_tensor = inputs[3];
            if (sizes_tensor.getDataType() != TensorDataType::INT64)
            {
                return OperatorExecuteResult::DATA_TYPE_ERROR;
            }

            sizes = std::vector<int64_t>(sizes_tensor.data<int64_t>(), sizes_tensor.data<int64_t>() + sizes_tensor.getNumElements());
            has_sizes = true;
        }

        if (has_scales == has_sizes)
        {
            return OperatorExecuteResult::ATTRIBUTE_ERROR; // One of 'scales' or 'sizes' must be specified, not both
        }

        // Handle different data types using a switch-case
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
