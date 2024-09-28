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
    __global__ void conv_kernel(const T *__restrict__ input_data, const size_t *__restrict__ input_dims, const size_t *__restrict__ input_strides,
                                const T *__restrict__ filter_data, const size_t *__restrict__ filter_dims, const size_t *__restrict__ filter_strides,
                                T *__restrict__ output_data, const size_t *__restrict__ output_dims, const size_t *__restrict__ output_strides,
                                size_t num_elements, size_t input_ndims, size_t filter_ndims, size_t output_ndims,
                                int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int group)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;

        size_t output_indices[MAX_DIMS];
        size_t tmp = idx;
        for (int i = output_ndims - 1; i >= 0; --i)
        {
            output_indices[i] = tmp % output_dims[i];
            tmp /= output_dims[i];
        }

        // Calculate the starting position for the convolution
        int out_h = output_indices[output_ndims - 2];
        int out_w = output_indices[output_ndims - 1];

        int in_h_start = out_h * stride_h - pad_h;
        int in_w_start = out_w * stride_w - pad_w;

        // Initialize the output value
        T output_val = 0;

        // Determine the input and output channels based on groups
        int out_channel = output_indices[1];
        int input_channel_start = (out_channel / (filter_dims[0] / group)) * (filter_dims[1]);

        // Perform the convolution operation
        for (size_t c = 0; c < filter_dims[1]; ++c)
        {
            for (size_t kh = 0; kh < filter_dims[2]; ++kh)
            {
                for (size_t kw = 0; kw < filter_dims[3]; ++kw)
                {
                    int in_h = in_h_start + kh * dilation_h;
                    int in_w = in_w_start + kw * dilation_w;

                    // Check for boundary conditions
                    if (in_h >= 0 && in_h < input_dims[input_ndims - 2] &&
                        in_w >= 0 && in_w < input_dims[input_ndims - 1])
                    {
                        size_t input_idx = 0;
                        input_idx += output_indices[0] * input_strides[0];         // Batch dimension
                        input_idx += (input_channel_start + c) * input_strides[1]; // Channel dimension adjusted for group
                        input_idx += in_h * input_strides[input_ndims - 2];
                        input_idx += in_w * input_strides[input_ndims - 1];

                        size_t filter_idx = 0;
                        filter_idx += out_channel * filter_strides[0]; // Output channel
                        filter_idx += c * filter_strides[1];
                        filter_idx += kh * filter_strides[2];
                        filter_idx += kw * filter_strides[3];

                        output_val += input_data[input_idx] * filter_data[filter_idx];
                    }
                }
            }
        }

        // Set the computed value to the output
        output_data[idx] = output_val;
    }

    OperatorExecuteResult ConvOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                    const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        const Tensor &input = inputs[0];
        const Tensor &filter = inputs[1];
        Tensor *output = outputs[0];

        // Extract attributes with default values
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

        TensorDataType dtype = input.getDataType();
        size_t num_elements_output = output->getNumElements();

        const void *input_data = input.getBuffer()->getDataPointer();
        const void *filter_data = filter.getBuffer()->getDataPointer();
        void *output_data = output->getBuffer()->getDataPointer();

        const auto &input_dims = input.getDims();
        const auto &filter_dims = filter.getDims();
        const auto &output_dims = output->getDims();
        const auto &input_strides = input.getStrides();
        const auto &filter_strides = filter.getStrides();
        const auto &output_strides = output->getStrides();

        size_t input_ndims = input_dims.size();
        size_t filter_ndims = filter_dims.size();
        size_t output_ndims = output_dims.size();

        size_t *d_input_dims = input.d_getDims();
        size_t *d_filter_dims = filter.d_getDims();
        size_t *d_output_dims = output->d_getDims();

        size_t *d_input_strides = input.d_getStrides();
        size_t *d_filter_strides = filter.d_getStrides();
        size_t *d_output_strides = output->d_getStrides();

        // Extract stride, padding, and dilation values
        int stride_h = static_cast<int>(strides[0]);
        int stride_w = static_cast<int>(strides[1]);
        int pad_h = static_cast<int>(pads[0]);
        int pad_w = static_cast<int>(pads[1]);
        int dilation_h = static_cast<int>(dilations[0]);
        int dilation_w = static_cast<int>(dilations[1]);

        dim3 gridSize((num_elements_output + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(conv_kernel<float>, gridSize, blockSize, 0, 0,
                                                    static_cast<const float *>(input_data), d_input_dims, d_input_strides,
                                                    static_cast<const float *>(filter_data), d_filter_dims, d_filter_strides,
                                                    static_cast<float *>(output_data), d_output_dims, d_output_strides,
                                                    num_elements_output, input_ndims, filter_ndims, output_ndims,
                                                    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group));
            break;
        case TensorDataType::FLOAT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(conv_kernel<double>, gridSize, blockSize, 0, 0,
                                                    static_cast<const double *>(input_data), d_input_dims, d_input_strides,
                                                    static_cast<const double *>(filter_data), d_filter_dims, d_filter_strides,
                                                    static_cast<double *>(output_data), d_output_dims, d_output_strides,
                                                    num_elements_output, input_ndims, filter_ndims, output_ndims,
                                                    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group));
            break;
        case TensorDataType::INT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(conv_kernel<int64_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int64_t *>(input_data), d_input_dims, d_input_strides,
                                                    static_cast<const int64_t *>(filter_data), d_filter_dims, d_filter_strides,
                                                    static_cast<int64_t *>(output_data), d_output_dims, d_output_strides,
                                                    num_elements_output, input_ndims, filter_ndims, output_ndims,
                                                    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group));
            break;
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        hipErrorCheck(hipDeviceSynchronize());

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif
