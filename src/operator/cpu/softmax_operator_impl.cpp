#include "operator/operators.hpp"
#include <cmath>

namespace CPU_OP
{
    template <typename T>
    OperatorExecuteResult executeSoftmax(const std::vector<Tensor> &inputs, Tensor *output, int64_t axis)
    {
        const Tensor &input = inputs[0];
        const T *input_data = static_cast<const T *>(input.getDataPointer());
        T *output_data = static_cast<T *>(output->getDataPointer());

        const std::vector<size_t> &input_shape = input.getDims();
        size_t rank = input_shape.size();

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

                T max_val = input_data[offset];
                for (size_t i = 1; i < axis_size; ++i)
                {
                    size_t idx = offset + i * inner_size;
                    if (input_data[idx] > max_val)
                    {
                        max_val = input_data[idx];
                    }
                }

                T sum = 0;
                for (size_t i = 0; i < axis_size; ++i)
                {
                    size_t idx = offset + i * inner_size;
                    output_data[idx] = std::exp(input_data[idx] - max_val);
                    sum = sum + output_data[idx];
                }

                for (size_t i = 0; i < axis_size; ++i)
                {
                    size_t idx = offset + i * inner_size;
                    output_data[idx] = output_data[idx] / sum;
                }
            }
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult SoftmaxOperatorImpl::execute(const std::vector<Tensor> &inputs,
                                                       std::vector<Tensor *> &outputs,
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        const Tensor &input = inputs[0];
        Tensor *output = outputs[0];

        TensorDataType dataType = input.getDataType();

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

        switch (dataType)
        {
        case TensorDataType::FLOAT32:
            return executeSoftmax<float>(inputs, output, axis);
        case TensorDataType::FLOAT64:
            return executeSoftmax<double>(inputs, output, axis);
        case TensorDataType::FLOAT16:
            return executeSoftmax<half_t>(inputs, output, axis);

        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }
}
