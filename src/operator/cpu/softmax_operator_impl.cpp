#include "operator/operators.hpp"
#include <cmath>

namespace CPU_OP
{

    OperatorExecuteResult SoftmaxOperatorImpl::execute(const std::vector<Tensor> &inputs,
                                                       std::vector<Tensor *> &outputs,
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {

        if (inputs.size() != 1)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        if (outputs.size() != 1 || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const Tensor &input = inputs[0];
        Tensor *output = outputs[0];

        TensorDataType dataType = input.getDataType();
        if (dataType != TensorDataType::FLOAT32)
        {
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }

        int64_t axis = -1;
        if (attributes.count("axis"))
        {
            axis = std::get<int64_t>(attributes.at("axis"));
        }

        const std::vector<size_t> &input_shape = input.getDims();
        size_t rank = input_shape.size();
        if (axis < 0)
        {
            axis += rank;
        }
        if (axis < 0 || static_cast<size_t>(axis) >= rank)
        {
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        }

        output->reshape(input_shape);
        output->setDataType(TensorDataType::FLOAT32);

        size_t num_elements = input.getNumElements();
        if (!output->data<float>() || output->getNumElements() != num_elements)
        {
            output->allocateBuffer(TensorDataType::FLOAT32, num_elements);
        }

        const float *input_data = input.data<float>();
        float *output_data = output->data<float>();

        if (!output_data)
        {
            return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
        }

        size_t outer_size = 1;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i)
        {
            outer_size *= input_shape[i];
        }

        size_t axis_size = input_shape[axis];

        size_t inner_size = 1;
        for (size_t i = static_cast<size_t>(axis) + 1; i < rank; ++i)
        {
            inner_size *= input_shape[i];
        }

        for (size_t outer = 0; outer < outer_size; ++outer)
        {
            for (size_t inner = 0; inner < inner_size; ++inner)
            {

                size_t offset = outer * axis_size * inner_size + inner;

                float max_val = input_data[offset];
                for (size_t i = 1; i < axis_size; ++i)
                {
                    size_t idx = offset + i * inner_size;
                    if (input_data[idx] > max_val)
                    {
                        max_val = input_data[idx];
                    }
                }

                float sum = 0.0f;
                for (size_t i = 0; i < axis_size; ++i)
                {
                    size_t idx = offset + i * inner_size;
                    output_data[idx] = std::exp(input_data[idx] - max_val);
                    sum += output_data[idx];
                }

                for (size_t i = 0; i < axis_size; ++i)
                {
                    size_t idx = offset + i * inner_size;
                    output_data[idx] /= sum;
                }
            }
        }

        return OperatorExecuteResult::SUCCESS;
    }
}
