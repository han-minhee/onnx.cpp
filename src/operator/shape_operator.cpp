#include "operator/operators.hpp"
#include <iostream>

// Implement inferOutputShapes and inferOutputDataTypes
std::vector<std::vector<size_t>> ShapeOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                  const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    // Validate the number of inputs
    if (inputs.size() != 1)
    {
        return {};
    }

    const Tensor &input_tensor = inputs[0];
    const std::vector<size_t> &input_shape = input_tensor.getDims();
    size_t rank = input_shape.size();

    // Get the 'start' attribute, default is 0
    int64_t start = 0;
    if (attributes.find("start") != attributes.end())
    {
        start = std::get<int64_t>(attributes.at("start"));
    }

    // Get the 'end' attribute, default is rank (all dimensions included)
    int64_t end = static_cast<int64_t>(rank);
    if (attributes.find("end") != attributes.end())
    {
        end = std::get<int64_t>(attributes.at("end"));
    }

    // Adjust negative indices
    if (start < 0)
    {
        start += static_cast<int64_t>(rank);
    }
    if (end < 0)
    {
        end += static_cast<int64_t>(rank);
    }

    // Clamp start and end to be within the valid range [0, rank]
    start = static_cast<int64_t>(std::max(0, std::min(static_cast<int>(start), static_cast<int>(rank))));
    end = static_cast<int64_t>(std::max(0, std::min(static_cast<int>(end), static_cast<int>(rank))));

    // Compute the output shape slice
    std::vector<size_t> output_shape_slice;
    for (int i = start; i < end; ++i)
    {
        output_shape_slice.push_back(static_cast<int64_t>(input_shape[i]));
    }

    return {output_shape_slice};
}

std::vector<TensorDataType> ShapeOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return {TensorDataType::INT64};
}

OperatorExecuteResult ShapeOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                             const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    // Validate the number of inputs
    if (inputs.size() != 1)
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    // Validate the output tensor
    if (outputs.empty() || outputs[0] == nullptr)
    {
        return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
    }

    const Tensor &input_tensor = inputs[0];
    Tensor *output_tensor = outputs[0];

    // Retrieve the shape of the input tensor
    const std::vector<size_t> &input_shape = input_tensor.getDims();
    size_t rank = input_shape.size();

    // Get the 'start' attribute, default is 0
    int64_t start = 0;
    if (attributes.find("start") != attributes.end())
    {
        start = std::get<int64_t>(attributes.at("start"));
    }

    // Get the 'end' attribute, default is rank (all dimensions included)
    int64_t end = static_cast<int64_t>(rank);
    if (attributes.find("end") != attributes.end())
    {
        end = std::get<int64_t>(attributes.at("end"));
    }

    // Adjust negative indices
    if (start < 0)
    {
        start += static_cast<int64_t>(rank);
    }
    if (end < 0)
    {
        end += static_cast<int64_t>(rank);
    }

    // Clamp start and end to be within the valid range [0, rank]
    start = static_cast<int64_t>(std::max(0, std::min(static_cast<int>(start), static_cast<int>(rank))));
    end = static_cast<int64_t>(std::max(0, std::min(static_cast<int>(end), static_cast<int>(rank))));

    // Compute the output shape slice
    std::vector<int64_t> output_shape_slice;
    for (int i = start; i < end; ++i)
    {
        output_shape_slice.push_back(static_cast<int64_t>(input_shape[i]));
    }

    // Allocate memory for the output tensor
    size_t output_num_elements = output_shape_slice.size();
    int64_t *output_data = new (std::nothrow) int64_t[output_num_elements];
    if (!output_data)
    {
        return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
    }

    // Copy the sliced shape data to the output tensor
    std::copy(output_shape_slice.begin(), output_shape_slice.end(), output_data);

    // Set the data pointer and shape of the output tensor
    std::vector<size_t> output_tensor_shape = {output_num_elements}; // 1D tensor with length of the slice

    output_tensor->setDataType(TensorDataType::INT64);
    output_tensor->setDataPointer<int64_t>(output_data, output_tensor_shape);

    return OperatorExecuteResult::SUCCESS;
}
