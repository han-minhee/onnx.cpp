#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

#define BLOCK_SIZE 256
#define MAX_DIMS 8

namespace HIP_OP
{
    template <typename T>
    __device__ float compute_x_original_device(float x_resized, float scale, const int coordinate_transformation_mode,
                                               size_t input_size, size_t output_size)
    {
        if (coordinate_transformation_mode == 0 || coordinate_transformation_mode == 1) // "half_pixel" or "pytorch_half_pixel"
        {
            return (output_size > 1) ? (x_resized + 0.5f) / scale - 0.5f : 0;
        }
        else if (coordinate_transformation_mode == 2) // "asymmetric"
        {
            return x_resized / scale;
        }
        else if (coordinate_transformation_mode == 3) // "align_corners"
        {
            return (output_size == 1) ? 0 : x_resized * (input_size - 1) / (output_size - 1);
        }
        return (x_resized + 0.5f) / scale - 0.5f; // Default behavior
    }

    template <typename T>
    __device__ size_t compute_x_input_device(float x_original, const int nearest_mode, size_t input_size)
    {
        float x_nearest = (nearest_mode == 0)   ? floor(x_original + 0.5f) // "round_prefer_floor"
                          : (nearest_mode == 1) ? ceil(x_original - 0.5f)  // "round_prefer_ceil"
                          : (nearest_mode == 2) ? floor(x_original)        // "floor"
                                                : ceil(x_original);        // default ceil

        return static_cast<size_t>(max(0.0f, min(x_nearest, static_cast<float>(input_size - 1))));
    }

    template <typename T>
    __global__ void resize_kernel(const T *__restrict__ input_data, const size_t *__restrict__ input_shape,
                                  const size_t *__restrict__ input_strides, T *__restrict__ output_data,
                                  const size_t *__restrict__ output_shape, const size_t *__restrict__ output_strides,
                                  const float *__restrict__ scales, int coordinate_transformation_mode, int nearest_mode,
                                  size_t num_output_elements, size_t num_dims)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_output_elements)
            return;

        size_t indices[MAX_DIMS] = {0};
        size_t tmp = idx;
        for (int i = num_dims - 1; i >= 0; --i)
        {
            indices[i] = tmp % output_shape[i];
            tmp /= output_shape[i];
        }

        size_t input_index = 0;
        for (size_t i = 0; i < num_dims; ++i)
        {
            float x_orig = compute_x_original_device<T>(static_cast<float>(indices[i]), scales[i],
                                                        coordinate_transformation_mode, input_shape[i], output_shape[i]);
            size_t x_input = compute_x_input_device<T>(x_orig, nearest_mode, input_shape[i]);
            input_index += x_input * input_strides[i];
        }

        size_t output_index = 0;
        for (size_t i = 0; i < num_dims; ++i)
        {
            output_index += indices[i] * output_strides[i];
        }

        output_data[output_index] = input_data[input_index];
    }

    OperatorExecuteResult ResizeOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                      const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
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
            scales.assign(scales_tensor.data<float>(), scales_tensor.data<float>() + scales_tensor.getNumElements());
        }

        if (has_sizes)
        {
            const Tensor &sizes_tensor = inputs[3];
            sizes.assign(sizes_tensor.data<int64_t>(), sizes_tensor.data<int64_t>() + sizes_tensor.getNumElements());
        }

        if (has_scales == has_sizes)
            return OperatorExecuteResult::ATTRIBUTE_ERROR;

        const void *input_data = input_tensor.getBuffer()->getDataPointer();
        void *output_data = output_tensor->getBuffer()->getDataPointer();

        const auto &input_shape = input_tensor.getDims();
        const auto &output_shape = output_tensor->getDims();
        const auto &input_strides = input_tensor.getStrides();
        const auto &output_strides = output_tensor->getStrides();

        size_t num_output_elements = output_tensor->getNumElements();
        size_t num_dims = output_shape.size();

        // Copy data to device
        size_t *d_input_shape = input_tensor.d_getDims();
        size_t *d_output_shape = output_tensor->d_getDims();
        size_t *d_input_strides = input_tensor.d_getStrides();
        size_t *d_output_strides = output_tensor->d_getStrides();

        float *d_scales = nullptr;
        if (has_scales)
        {
            hipErrorCheck(hipMalloc(&d_scales, scales.size() * sizeof(float)));
            hipErrorCheck(hipMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), hipMemcpyHostToDevice));
        }

        // Convert string options to integers for use in the device
        int coordinate_transformation_mode_int = (coordinate_transformation_mode == "half_pixel")           ? 0
                                                 : (coordinate_transformation_mode == "pytorch_half_pixel") ? 1
                                                 : (coordinate_transformation_mode == "asymmetric")         ? 2
                                                                                                            : 3; // align_corners or default

        int nearest_mode_int = (nearest_mode == "round_prefer_floor")  ? 0
                               : (nearest_mode == "round_prefer_ceil") ? 1
                               : (nearest_mode == "floor")             ? 2
                                                                       : 3; // ceil

        dim3 gridSize((num_output_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        switch (input_tensor.getDataType())
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(resize_kernel<float>, gridSize, blockSize, 0, 0,
                                                    static_cast<const float *>(input_data), d_input_shape, d_input_strides,
                                                    static_cast<float *>(output_data), d_output_shape, d_output_strides,
                                                    d_scales, coordinate_transformation_mode_int, nearest_mode_int,
                                                    num_output_elements, num_dims));
            break;
        case TensorDataType::INT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(resize_kernel<int32_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int32_t *>(input_data), d_input_shape, d_input_strides,
                                                    static_cast<int32_t *>(output_data), d_output_shape, d_output_strides,
                                                    d_scales, coordinate_transformation_mode_int, nearest_mode_int,
                                                    num_output_elements, num_dims));
            break;
        case TensorDataType::INT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(resize_kernel<int64_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int64_t *>(input_data), d_input_shape, d_input_strides,
                                                    static_cast<int64_t *>(output_data), d_output_shape, d_output_strides,
                                                    d_scales, coordinate_transformation_mode_int, nearest_mode_int,
                                                    num_output_elements, num_dims));
            break;
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        hipErrorCheck(hipDeviceSynchronize());

        if (d_scales != nullptr)
        {
            hipErrorCheck(hipFree(d_scales));
        }

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif
