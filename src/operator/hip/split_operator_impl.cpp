#ifdef USE_HIP
#include "operator/operators.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 256

namespace HIP_OP
{
    template <typename T>
    __global__ void split_kernel(const T *__restrict__ input_data, T **output_data,
                                 const size_t *input_dims, const size_t *output_sizes,
                                 size_t num_splits, size_t axis, size_t outer_dim, size_t inner_dim, size_t input_axis_dim)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        size_t offset = 0;
        for (size_t i = 0; i < num_splits; ++i)
        {
            size_t split_size = output_sizes[i];

            if (idx < outer_dim * split_size * inner_dim)
            {

                size_t outer = idx / (split_size * inner_dim);
                size_t split_idx = (idx / inner_dim) % split_size;
                size_t inner = idx % inner_dim;

                size_t input_index = outer * input_axis_dim * inner_dim + (offset + split_idx) * inner_dim + inner;

                output_data[i][idx] = input_data[input_index];
                break;
            }

            idx -= outer_dim * split_size * inner_dim;
            offset += split_size;
        }
    }

    OperatorExecuteResult SplitOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        std::vector<std::vector<size_t>> output_shapes;
        for (const Tensor *output : outputs)
        {
            if (output == nullptr)
            {
                return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
            }
            if (output->getDims().empty())
            {
                return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
            }
            output_shapes.push_back(output->getDims());
        }

        const Tensor &input = inputs.at(0);

        int64_t axis = 0;
        if (attributes.count("axis"))
        {
            axis = std::get<int64_t>(attributes.at("axis"));
        }

        size_t rank = input.getNDim();
        if (axis < 0)
        {
            axis += rank;
        }

        std::vector<int64_t> split_sizes(output_shapes.size());
        for (size_t i = 0; i < output_shapes.size(); ++i)
        {
            split_sizes[i] = static_cast<int64_t>(output_shapes[i][axis]);
        }

        size_t outer_dim = 1;
        for (size_t dim = 0; dim < static_cast<size_t>(axis); ++dim)
        {
            outer_dim *= input.getDims()[dim];
        }

        size_t inner_dim = 1;
        for (size_t dim = static_cast<size_t>(axis) + 1; dim < rank; ++dim)
        {
            inner_dim *= input.getDims()[dim];
        }

        size_t input_axis_dim = input.getDims()[axis];

        const void *input_data = input.getDataPointer();

        void **d_output_data;
        hipErrorCheck(hipMalloc(&d_output_data, outputs.size() * sizeof(void *)));

        std::vector<void *> h_output_data(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            h_output_data[i] = outputs[i]->getDataPointer();
        }
        hipErrorCheck(hipMemcpy(d_output_data, h_output_data.data(), outputs.size() * sizeof(void *), hipMemcpyHostToDevice));

        size_t *d_split_sizes;
        hipErrorCheck(hipMalloc(&d_split_sizes, split_sizes.size() * sizeof(size_t)));
        hipErrorCheck(hipMemcpy(d_split_sizes, split_sizes.data(), split_sizes.size() * sizeof(size_t), hipMemcpyHostToDevice));

        size_t num_elements = 0;
        for (const auto &size : split_sizes)
        {
            num_elements += size * outer_dim * inner_dim;
        }

        dim3 gridSize(CeilDiv(num_elements, BLOCK_SIZE));
        dim3 blockSize(BLOCK_SIZE);

        switch (input.getDataType())
        {
        case TensorDataType::FLOAT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(split_kernel<float>, gridSize, blockSize, 0, 0,
                                                    static_cast<const float *>(input_data),
                                                    reinterpret_cast<float **>(d_output_data),
                                                    input.getDims().data(), d_split_sizes,
                                                    split_sizes.size(), axis, outer_dim, inner_dim, input_axis_dim));
            break;
        case TensorDataType::FLOAT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(split_kernel<double>, gridSize, blockSize, 0, 0,
                                                    static_cast<const double *>(input_data),
                                                    reinterpret_cast<double **>(d_output_data),
                                                    input.getDims().data(), d_split_sizes,
                                                    split_sizes.size(), axis, outer_dim, inner_dim, input_axis_dim));
            break;
        case TensorDataType::INT32:
            hipKernelLaunchCheck(hipLaunchKernelGGL(split_kernel<int32_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int32_t *>(input_data),
                                                    reinterpret_cast<int32_t **>(d_output_data),
                                                    input.getDims().data(), d_split_sizes,
                                                    split_sizes.size(), axis, outer_dim, inner_dim, input_axis_dim));
            break;
        case TensorDataType::INT64:
            hipKernelLaunchCheck(hipLaunchKernelGGL(split_kernel<int64_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int64_t *>(input_data),
                                                    reinterpret_cast<int64_t **>(d_output_data),
                                                    input.getDims().data(), d_split_sizes,
                                                    split_sizes.size(), axis, outer_dim, inner_dim, input_axis_dim));
            break;
        case TensorDataType::INT8:
            hipKernelLaunchCheck(hipLaunchKernelGGL(split_kernel<int8_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const int8_t *>(input_data),
                                                    reinterpret_cast<int8_t **>(d_output_data),
                                                    input.getDims().data(), d_split_sizes,
                                                    split_sizes.size(), axis, outer_dim, inner_dim, input_axis_dim));
            break;
        case TensorDataType::UINT8:
            hipKernelLaunchCheck(hipLaunchKernelGGL(split_kernel<uint8_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const uint8_t *>(input_data),
                                                    reinterpret_cast<uint8_t **>(d_output_data),
                                                    input.getDims().data(), d_split_sizes,
                                                    split_sizes.size(), axis, outer_dim, inner_dim, input_axis_dim));
            break;
        case TensorDataType::FLOAT16:
            hipKernelLaunchCheck(hipLaunchKernelGGL(split_kernel<half_t>, gridSize, blockSize, 0, 0,
                                                    static_cast<const half_t *>(input_data),
                                                    reinterpret_cast<half_t **>(d_output_data),
                                                    input.getDims().data(), d_split_sizes,
                                                    split_sizes.size(), axis, outer_dim, inner_dim, input_axis_dim));
            break;
        default:
            hipErrorCheck(hipFree(d_split_sizes));
            hipErrorCheck(hipFree(d_output_data));
        }

        hipErrorCheck(hipFree(d_split_sizes));
        hipErrorCheck(hipFree(d_output_data));

        return OperatorExecuteResult::SUCCESS;
    }
}

#endif
