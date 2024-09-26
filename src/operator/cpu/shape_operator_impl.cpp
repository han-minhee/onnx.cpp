#include "operator/operators.hpp"
#include <iostream>

namespace CPU_OP
{
    template <typename T>
    OperatorExecuteResult executeShape(const Tensor &input_tensor, Tensor *output_tensor,
                                       int64_t start, int64_t end)
    {
        const std::vector<size_t> &input_shape = input_tensor.getDims();
        size_t rank = input_shape.size();

        if (start < 0)
        {
            start += static_cast<int64_t>(rank);
        }
        if (end < 0)
        {
            end += static_cast<int64_t>(rank);
        }

        start = static_cast<int64_t>(std::max(0, std::min(static_cast<int>(start), static_cast<int>(rank))));
        end = static_cast<int64_t>(std::max(0, std::min(static_cast<int>(end), static_cast<int>(rank))));

        std::vector<int64_t> output_shape_slice;
        for (int i = start; i < end; ++i)
        {
            output_shape_slice.push_back(static_cast<int64_t>(input_shape[i]));
        }

        size_t output_num_elements = output_shape_slice.size();
        std::vector<size_t> output_tensor_shape = {output_num_elements};

        output_tensor->reshape(output_tensor_shape);
        output_tensor->setDataType(TensorDataType::INT64);

        if (!output_tensor->data<int64_t>() || output_tensor->getNumElements() != output_num_elements)
        {
            output_tensor->allocateBuffer(TensorDataType::INT64, output_num_elements);
        }

        int64_t *output_data = output_tensor->data<int64_t>();
        if (!output_data)
        {
            return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
        }

        std::copy(output_shape_slice.begin(), output_shape_slice.end(), output_data);

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult ShapeOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {

        if (inputs.size() != 1)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        if (outputs.empty() || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const Tensor &input_tensor = inputs[0];
        Tensor *output_tensor = outputs[0];

        int64_t start = 0;
        if (attributes.find("start") != attributes.end())
        {
            start = std::get<int64_t>(attributes.at("start"));
        }

        int64_t end = static_cast<int64_t>(input_tensor.getDims().size());
        if (attributes.find("end") != attributes.end())
        {
            end = std::get<int64_t>(attributes.at("end"));
        }

        switch (input_tensor.getDataType())
        {
        case TensorDataType::INT32:
            return executeShape<int32_t>(input_tensor, output_tensor, start, end);
        case TensorDataType::INT64:
            return executeShape<int64_t>(input_tensor, output_tensor, start, end);
        case TensorDataType::FLOAT32:
            return executeShape<float>(input_tensor, output_tensor, start, end);
        case TensorDataType::FLOAT64:
            return executeShape<double>(input_tensor, output_tensor, start, end);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
}
