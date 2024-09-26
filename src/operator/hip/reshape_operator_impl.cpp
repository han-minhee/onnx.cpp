#ifdef USE_HIP
#include <hip/hip_runtime_api.h>

#include "operator/operators.hpp"
#include <iostream>

namespace HIP_OP
{
    template <typename T>
    OperatorExecuteResult executeReshape(const Tensor &input_tensor, const Tensor &shape_tensor, Tensor *output_tensor,
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
                if (allowzero)
                {
                    output_shape[i] = 0;
                }
                else
                {
                    output_shape[i] = input_tensor.getDims()[i];
                }
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

        std::copy(input_data, input_data + input_num_elements, output_data);

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult ReshapeOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {

        if (inputs.size() != 2)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        if (outputs.empty() || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const Tensor &input_tensor = inputs[0];
        const Tensor &shape_tensor = inputs[1];
        Tensor *output_tensor = outputs[0];

        if (shape_tensor.getDataType() != TensorDataType::INT64 || shape_tensor.getNDim() != 1)
        {
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        bool allowzero = false;
        if (attributes.find("allowzero") != attributes.end())
        {
            allowzero = static_cast<int64_t>(std::get<int64_t>(attributes.at("allowzero"))) != 0;
        }

        switch (input_tensor.getDataType())
        {
        case TensorDataType::FLOAT32:
            return executeReshape<float>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::FLOAT64:
            return executeReshape<double>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::INT32:
            return executeReshape<int32_t>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::INT64:
            return executeReshape<int64_t>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::INT8:
            return executeReshape<int8_t>(input_tensor, shape_tensor, output_tensor, allowzero);
        case TensorDataType::UINT8:
            return executeReshape<uint8_t>(input_tensor, shape_tensor, output_tensor, allowzero);
        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }
}

#endif