#include "operator/operators.hpp"
#include <iostream>
#include <cmath>

namespace CPU_OP
{
    template <typename T>
    OperatorExecuteResult executeSigmoid(const std::vector<Tensor> &inputs, Tensor *output)
    {

        const Tensor &input = inputs[0];
        const T *input_data = static_cast<const T *>(input.getDataPointer());
        T *output_data = static_cast<T *>(output->getDataPointer());

        size_t num_elements = input.getNumElements();

        for (size_t i = 0; i < num_elements; i++)
        {
            output_data[i] = 1.0f / (1.0f + exp(-input_data[i]));
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult SigmoidOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        const TensorDataType dataType = inputs[0].getDataType();
        Tensor *output = outputs[0];

        switch (dataType)
        {
        case TensorDataType::FLOAT32:
            return executeSigmoid<float>(inputs, output);
        case TensorDataType::FLOAT64:
            return executeSigmoid<double>(inputs, output);
        case TensorDataType::FLOAT16:
            return executeSigmoid<half_t>(inputs, output);

        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }
}
