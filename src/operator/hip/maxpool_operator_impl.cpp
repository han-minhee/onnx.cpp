#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#include "utils.hpp"

#define BLOCK_SIZE 256

namespace HIP_OP
{

    template <typename T>
    __global__ void maxpool_kernel(const T *__restrict__ input_data, const size_t *__restrict__ input_dims, const size_t *__restrict__ input_strides,
                                   T *__restrict__ output_data, const size_t *__restrict__ output_dims, const size_t *__restrict__ output_strides,
                                   size_t num_elements, size_t input_ndims, size_t output_ndims,
                                   int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
                                   int dilation_h, int dilation_w, int ceil_mode)
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

        int out_h = output_indices[output_ndims - 2];
        int out_w = output_indices[output_ndims - 1];

        int in_h_start = out_h * stride_h - pad_h;
        int in_w_start = out_w * stride_w - pad_w;

        T max_val = -std::numeric_limits<T>::infinity();

        for (int kh = 0; kh < kernel_h; ++kh)
        {
            for (int kw = 0; kw < kernel_w; ++kw)
            {
                int in_h = in_h_start + kh * dilation_h;
                int in_w = in_w_start + kw * dilation_w;

                if (in_h >= 0 && in_h < input_dims[input_ndims - 2] &&
                    in_w >= 0 && in_w < input_dims[input_ndims - 1])
                {
                    size_t input_idx = 0;
                    input_idx += output_indices[0] * input_strides[0];
                    input_idx += output_indices[1] * input_strides[1];
                    input_idx += in_h * input_strides[input_ndims - 2];
                    input_idx += in_w * input_strides[input_ndims - 1];

                    max_val = max(max_val, input_data[input_idx]);
                }
            }
        }

        output_data[idx] = max_val;
    }

    OperatorExecuteResult MaxPoolOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        const Tensor &input = inputs[0];
        Tensor *output = outputs[0];

        std::string auto_pad = "NOTSET";
        int64_t ceil_mode = 0;
        std::vector<int64_t> dilations = {1, 1};
        std::vector<int64_t> kernel_shape;
        std::vector<int64_t> pads = {0, 0, 0, 0};
        int64_t storage_order = 0;
        std::vector<int64_t> strides = {1, 1};

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

        TensorDataType dtype = input.getDataType();
        size_t num_elements_output = output->getNumElements();

        const void *input_data = input.getBuffer()->getDataPointer();
        void *output_data = output->getBuffer()->getDataPointer();

        const auto &input_dims = input.getDims();
        const auto &output_dims = output->getDims();
        const auto &input_strides = input.getStrides();
        const auto &output_strides = output->getStrides();

        size_t input_ndims = input_dims.size();
        size_t output_ndims = output_dims.size();

        size_t *d_input_dims = input.d_getDims();
        size_t *d_output_dims = output->d_getDims();
        size_t *d_input_strides = input.d_getStrides();
        size_t *d_output_strides = output->d_getStrides();

        int kernel_h = static_cast<int>(kernel_shape[0]);
        int kernel_w = static_cast<int>(kernel_shape[1]);
        int stride_h = static_cast<int>(strides[0]);
        int stride_w = static_cast<int>(strides[1]);
        int pad_h = static_cast<int>(pads[0]);
        int pad_w = static_cast<int>(pads[1]);
        int dilation_h = static_cast<int>(dilations[0]);
        int dilation_w = static_cast<int>(dilations[1]);

        dim3 gridSize(CeilDiv(num_elements_output, BLOCK_SIZE));
        dim3 blockSize(BLOCK_SIZE);

        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(maxpool_kernel<float>, gridSize, blockSize, 0, 0,
                                                    static_cast<const float *>(input_data), d_input_dims, d_input_strides,
                                                    static_cast<float *>(output_data), d_output_dims, d_output_strides,
                                                    num_elements_output, input_ndims, output_ndims,
                                                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode));
            break;
        case TensorDataType::FLOAT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(maxpool_kernel<double>, gridSize, blockSize, 0, 0,
                                                    static_cast<const double *>(input_data), d_input_dims, d_input_strides,
                                                    static_cast<double *>(output_data), d_output_dims, d_output_strides,
                                                    num_elements_output, input_ndims, output_ndims,
                                                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode));
            break;
        case TensorDataType::INT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(maxpool_kernel<int64_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int64_t *>(input_data), d_input_dims, d_input_strides,
                                                    static_cast<int64_t *>(output_data), d_output_dims, d_output_strides,
                                                    num_elements_output, input_ndims, output_ndims,
                                                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode));
            break;
        case TensorDataType::INT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(maxpool_kernel<int32_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int32_t *>(input_data), d_input_dims, d_input_strides,
                                                    static_cast<int32_t *>(output_data), d_output_dims, d_output_strides,
                                                    num_elements_output, input_ndims, output_ndims,
                                                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode));
            break;

        case TensorDataType::FLOAT16:
            hipKernelLaunchCheck(hipLaunchKernelGGL(maxpool_kernel<half_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const half_t *>(input_data), d_input_dims, d_input_strides,
                                                    static_cast<half_t *>(output_data), d_output_dims, d_output_strides,
                                                    num_elements_output, input_ndims, output_ndims,
                                                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode));
            break;

        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        hipErrorCheck(hipDeviceSynchronize());

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif
