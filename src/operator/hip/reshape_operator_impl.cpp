#ifdef USE_HIP
#include "operator/operators.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 256

namespace HIP_OP
{
    template <typename T>
    __global__ void reshape_copy_kernel(const T *__restrict__ input_data, T *__restrict__ output_data, size_t num_elements)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements)
        {
            output_data[idx] = input_data[idx];
        }
    }

    template <typename T>
    OperatorExecuteResult executeReshapeHIP(const Tensor &input_tensor, const Tensor &shape_tensor, Tensor *output_tensor,
                                            bool allowzero)
    {
        const int64_t *shape_data = shape_tensor.data<int64_t>();
        size_t shape_size = shape_tensor.getNumElements();
        size_t input_num_elements = input_tensor.getNumElements();

        std::vector<size_t> output_shape(shape_size);
        size_t inferred_dimension = 1;
        int64_t minus_one_pos = -1;

        for (size_t i = 0; i < shape_size; ++i)
        {
            int64_t dim = shape_data[i];
            if (dim == -1)
            {
                if (minus_one_pos != -1)
                {
                    return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
                }
                minus_one_pos = static_cast<int64_t>(i);
            }
            else if (dim == 0)
            {
                output_shape[i] = allowzero ? 0 : input_tensor.getDims()[i];
            }
            else if (dim > 0)
            {
                output_shape[i] = static_cast<size_t>(dim);
                inferred_dimension *= output_shape[i];
            }
            else
            {
                return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
            }
        }

        if (minus_one_pos != -1)
        {
            if (input_num_elements % inferred_dimension != 0)
            {
                return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
            }
            output_shape[minus_one_pos] = input_num_elements / inferred_dimension;
        }

        size_t output_num_elements = 1;
        for (size_t dim : output_shape)
        {
            output_num_elements *= dim;
        }

        if (output_num_elements != input_num_elements)
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        output_tensor->reshape(output_shape);
        output_tensor->setDataType(input_tensor.getDataType());

        if (!output_tensor->data<T>() || output_tensor->getNumElements() != input_num_elements)
        {
            output_tensor->allocateBuffer(input_tensor.getDataType(), input_num_elements);
        }

        const T *input_data = input_tensor.data<T>();
        T *output_data = output_tensor->data<T>();

        if (!input_data || !output_data)
        {
            return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
        }

        dim3 gridSize(CeilDiv(input_num_elements, BLOCK_SIZE));
        dim3 blockSize(BLOCK_SIZE);

        hipKernelLaunchCheck(hipLaunchKernelGGL(reshape_copy_kernel<T>, gridSize, blockSize, 0, 0,
                                                input_data, output_data, input_num_elements));

        hipErrorCheck(hipDeviceSynchronize());

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult ReshapeOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        const Tensor &input_tensor = inputs[0];
        const Tensor &shape_tensor = inputs[1];
        Tensor *output_tensor = outputs[0];

        bool allowzero = false;
        if (attributes.find("allowzero") != attributes.end())
        {
            allowzero = static_cast<int64_t>(std::get<int64_t>(attributes.at("allowzero"))) != 0;
        }

        switch (input_tensor.getDataType())
        {
        case TensorDataType::FLOAT32:
            return executeReshapeHIP<float>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::FLOAT64:
            return executeReshapeHIP<double>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::INT32:
            return executeReshapeHIP<int32_t>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::INT64:
            return executeReshapeHIP<int64_t>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::INT8:
            return executeReshapeHIP<int8_t>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::UINT8:
            return executeReshapeHIP<uint8_t>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::FLOAT16:
            return executeReshapeHIP<half_t>(input_tensor, shape_tensor, output_tensor, allowzero);
        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }
}

#endif
