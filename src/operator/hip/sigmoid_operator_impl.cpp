#ifdef USE_HIP
#include <hip/hip_runtime_api.h>
#include "operator/operators.hpp"
#include <iostream>
#include <cmath>

namespace HIP_OP
{
    template <typename T>
    OperatorExecuteResult executeSigmoid(const std::vector<Tensor> &inputs, Tensor *output,
                                         const std::vector<size_t> &input_strides,
                                         const std::vector<size_t> &output_strides,
                                         const std::vector<size_t> &output_shape)
    {
        size_t num_elements = 1;
        for (size_t dim : output_shape)
        {
            num_elements *= dim;
        }

        // Ensure that the output tensor is properly allocated and has the correct shape
        output->reshape(output_shape);
        output->setDataType(inputs[0].getDataType());

        if (!output->data<T>() || output->getNumElements() != num_elements)
        {
            output->allocateBuffer(inputs[0].getDataType(), num_elements);
        }

        T *output_data = output->data<T>(); // Get the buffer pointer
        if (!output_data)
        {
            return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
        }

        const T *input_data = inputs[0].data<T>();
        size_t num_dims = output_shape.size();

        for (size_t idx = 0; idx < num_elements; ++idx)
        {
            size_t remainder = idx;
            std::vector<size_t> indices(num_dims, 0);
            for (size_t dim = 0; dim < num_dims; ++dim)
            {
                indices[dim] = remainder / output_strides[dim];
                remainder = remainder % output_strides[dim];
            }

            size_t offset = 0;
            for (size_t dim = 0; dim < num_dims; ++dim)
            {
                size_t input_dim_size = inputs[0].getDims()[dim];
                size_t index = (input_dim_size == 1) ? 0 : indices[dim];
                offset += index * input_strides[dim];
            }

            T value = 1 / (1 + std::exp(-input_data[offset]));
            output_data[idx] = value;
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult SigmoidOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        if (inputs.size() != 1)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        const TensorDataType dataType = inputs[0].getDataType();

        std::vector<size_t> output_shape = inputs[0].getDims();
        if (output_shape.empty())
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        if (outputs.empty() || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        Tensor *output = outputs[0];

        size_t num_dims = output_shape.size();
        std::vector<size_t> input_strides(num_dims, 1);
        std::vector<size_t> output_strides(num_dims, 1);

        for (int i = num_dims - 2; i >= 0; --i)
        {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        const auto &input_shape = inputs[0].getDims();
        for (int i = num_dims - 2; i >= 0; --i)
        {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        switch (dataType)
        {
        case TensorDataType::FLOAT32:
            return executeSigmoid<float>(inputs, output, input_strides, output_strides, output_shape);
        case TensorDataType::FLOAT64:
            return executeSigmoid<double>(inputs, output, input_strides, output_strides, output_shape);

        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }
}

#endif // USE_HIP