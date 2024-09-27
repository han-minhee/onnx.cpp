#ifdef USE_HIP
#include "operator/operators.hpp"
#include "operator/hip/elementwise_operator.hpp"

#include <hip/hip_runtime.h>
#include <vector>

namespace HIP_OP
{

#define MAX_DIMS 8

    template <typename T>
    __global__ void elementwiseAddKernel(const T **input_data_ptrs,
                                         T *output_data,
                                         const size_t *input_strides,
                                         const size_t *input_shapes,
                                         const size_t *output_strides,
                                         const size_t *output_shape,
                                         size_t num_inputs,
                                         size_t num_dims,
                                         size_t num_elements)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_elements)
        {
            return;
        }

        size_t indices[MAX_DIMS] = {0};
        size_t remainder = idx;
        for (int dim = 0; dim < num_dims; ++dim)
        {
            size_t stride = output_strides[dim];
            indices[dim] = remainder / stride;
            remainder %= stride;
        }

        T result = static_cast<T>(0);
        for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx)
        {
            const T *input_data = input_data_ptrs[input_idx];
            const size_t *input_stride = &input_strides[input_idx * MAX_DIMS];
            const size_t *input_shape = &input_shapes[input_idx * MAX_DIMS];

            size_t input_offset = 0;
            for (size_t dim = 0; dim < num_dims; ++dim)
            {
                size_t dim_index = indices[dim];
                if (input_shape[dim] == 1)
                {
                    dim_index = 0;
                }
                input_offset += dim_index * input_stride[dim];
            }

            T operand = input_data[input_offset];
            if (input_idx == 0)
            {
                result = operand;
            }
            else
            {
                result += operand;
            }
        }

        output_data[idx] = result;
    }

    template <typename T>
    OperatorExecuteResult executeElementwiseOperation(const std::vector<Tensor> &inputs,
                                                      Tensor *output,
                                                      const std::vector<std::vector<size_t>> &input_strides,
                                                      const std::vector<size_t> &output_strides,
                                                      const std::vector<size_t> &output_shape,
                                                      std::plus<T> op)
    {
        const size_t num_elements = output->getNumElements();

        if (!output->getBuffer() || output->getNumElements() != num_elements)
        {
            output->allocateBuffer(output->getDataType(), num_elements);
        }

        size_t num_inputs = inputs.size();
        size_t num_dims = output_shape.size();

        std::vector<T *> d_input_data_ptrs(num_inputs);
        for (size_t idx = 0; idx < num_inputs; ++idx)
        {
            size_t input_num_elements = inputs[idx].getNumElements();
            T *d_input_data;
            hipMalloc(&d_input_data, input_num_elements * sizeof(T));
            hipMemcpy(d_input_data, inputs[idx].data<T>(), input_num_elements * sizeof(T), hipMemcpyHostToDevice);
            d_input_data_ptrs[idx] = d_input_data;
        }

        T *d_output_data;
        hipMalloc(&d_output_data, num_elements * sizeof(T));

        std::vector<size_t> h_input_strides(num_inputs * MAX_DIMS, 1);
        std::vector<size_t> h_input_shapes(num_inputs * MAX_DIMS, 1);

        for (size_t idx = 0; idx < num_inputs; ++idx)
        {
            const auto &strides = input_strides[idx];
            const auto &adjusted_shape = inputs[idx].getDims();
            size_t dim_offset = idx * MAX_DIMS;
            size_t dim_gap = MAX_DIMS - adjusted_shape.size();

            for (size_t dim = 0; dim < adjusted_shape.size(); ++dim)
            {
                h_input_strides[dim_offset + dim_gap + dim] = strides[dim];
                h_input_shapes[dim_offset + dim_gap + dim] = adjusted_shape[dim];
            }
        }

        std::vector<size_t> h_output_strides(MAX_DIMS, 1);
        std::vector<size_t> h_output_shape(MAX_DIMS, 1);
        size_t dim_gap = MAX_DIMS - num_dims;

        for (size_t dim = 0; dim < num_dims; ++dim)
        {
            h_output_strides[dim_gap + dim] = output_strides[dim];
            h_output_shape[dim_gap + dim] = output_shape[dim];
        }

        size_t *d_input_strides;
        hipMalloc(&d_input_strides, num_inputs * MAX_DIMS * sizeof(size_t));
        hipMemcpy(d_input_strides, h_input_strides.data(), num_inputs * MAX_DIMS * sizeof(size_t), hipMemcpyHostToDevice);

        size_t *d_input_shapes;
        hipMalloc(&d_input_shapes, num_inputs * MAX_DIMS * sizeof(size_t));
        hipMemcpy(d_input_shapes, h_input_shapes.data(), num_inputs * MAX_DIMS * sizeof(size_t), hipMemcpyHostToDevice);

        size_t *d_output_strides;
        hipMalloc(&d_output_strides, MAX_DIMS * sizeof(size_t));
        hipMemcpy(d_output_strides, h_output_strides.data(), MAX_DIMS * sizeof(size_t), hipMemcpyHostToDevice);

        size_t *d_output_shape;
        hipMalloc(&d_output_shape, MAX_DIMS * sizeof(size_t));
        hipMemcpy(d_output_shape, h_output_shape.data(), MAX_DIMS * sizeof(size_t), hipMemcpyHostToDevice);

        T **d_input_data_ptrs_array;
        hipMalloc(&d_input_data_ptrs_array, num_inputs * sizeof(T *));
        hipMemcpy(d_input_data_ptrs_array, d_input_data_ptrs.data(), num_inputs * sizeof(T *), hipMemcpyHostToDevice);

        size_t threads_per_block = 256;
        size_t blocks = (num_elements + threads_per_block - 1) / threads_per_block;

        hipLaunchKernelGGL(elementwiseAddKernel<T>, dim3(blocks), dim3(threads_per_block), 0, 0,
                           (const T **)d_input_data_ptrs_array, d_output_data,
                           d_input_strides, d_input_shapes,
                           d_output_strides, d_output_shape,
                           num_inputs, num_dims, num_elements);

        hipDeviceSynchronize();

        hipMemcpy(output->data<T>(), d_output_data, num_elements * sizeof(T), hipMemcpyDeviceToHost);

        for (size_t idx = 0; idx < num_inputs; ++idx)
        {
            hipFree(d_input_data_ptrs[idx]);
        }
        hipFree(d_input_data_ptrs_array);
        hipFree(d_input_strides);
        hipFree(d_input_shapes);
        hipFree(d_output_data);
        hipFree(d_output_strides);
        hipFree(d_output_shape);

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult AddOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        if (inputs.size() < 2)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        const TensorDataType dataType = inputs[0].getDataType();
        for (size_t i = 1; i < inputs.size(); i++)
        {
            if (inputs[i].getDataType() != dataType)
            {
                return OperatorExecuteResult::DATA_TYPE_ERROR;
            }
        }

        std::vector<std::vector<size_t>> input_shapes;
        for (const auto &tensor : inputs)
        {
            input_shapes.push_back(tensor.getDims());
        }

        Tensor *output = outputs[0];
        std::vector<size_t> output_shape = output->getDims();
        if (output_shape.empty())
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        if (outputs.empty() || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        size_t num_dims = output->getNDim();
        std::vector<size_t> output_strides = output->getStrides();
        std::vector<std::vector<size_t>> input_strides(inputs.size());
        for (size_t idx = 0; idx < inputs.size(); ++idx)
        {
            input_strides[idx] = compute_broadcast_strides(inputs[idx].getDims(), output_shape);
        }

        switch (dataType)
        {
        case TensorDataType::FLOAT32:
            return executeElementwiseOperation<float>(inputs, output, input_strides, output_strides, output_shape,
                                                      std::plus<float>());
        case TensorDataType::FLOAT64:
            return executeElementwiseOperation<double>(inputs, output, input_strides, output_strides, output_shape,
                                                       std::plus<double>());
        case TensorDataType::INT32:
            return executeElementwiseOperation<int32_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::plus<int32_t>());
        case TensorDataType::INT64:
            return executeElementwiseOperation<int64_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::plus<int64_t>());
        case TensorDataType::INT8:
            return executeElementwiseOperation<int8_t>(inputs, output, input_strides, output_strides, output_shape,
                                                       std::plus<int8_t>());
        case TensorDataType::UINT8:
            return executeElementwiseOperation<uint8_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::plus<uint8_t>());
        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }
};

#endif
