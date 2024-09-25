#include "operator/operators.hpp"
#include <cmath>

// implement inferOutputShapes and inferOutputDataTypes

std::vector<std::vector<size_t>> SoftmaxOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() != 1)
    {
        return {}; // Softmax requires exactly one input tensor
    }

    return {inputs.at(0).getDims()};
}

std::vector<TensorDataType> SoftmaxOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                  const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return {};
    }
    return {inputs.at(0).getDataType()};
}

OperatorExecuteResult SoftmaxOperator::execute(const std::vector<Tensor> &inputs,
                                               std::vector<Tensor *> &outputs,
                                               const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    // Check input tensor
    if (inputs.size() != 1)
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR; // Softmax requires exactly one input tensor
    }

    // Check output tensor
    if (outputs.size() != 1 || outputs[0] == nullptr)
    {
        return OperatorExecuteResult::OUTPUT_TENSOR_ERROR; // Softmax requires exactly one output tensor
    }

    const Tensor &input = inputs[0];
    Tensor *output = outputs[0];

    // Get data type and ensure it's supported
    TensorDataType dataType = input.getDataType();
    if (dataType != TensorDataType::FLOAT32)
    {
        return OperatorExecuteResult::UNSUPPORTED_OPERATION; // Only FLOAT32 is supported currently
    }

    // Get axis attribute (default is -1)
    int64_t axis = -1;
    if (attributes.count("axis"))
    {
        axis = std::get<int64_t>(attributes.at("axis"));
    }

    // Adjust negative axis
    const std::vector<size_t> &input_shape = input.getDims();
    size_t rank = input_shape.size();
    if (axis < 0)
    {
        axis += rank;
    }
    if (axis < 0 || static_cast<size_t>(axis) >= rank)
    {
        return OperatorExecuteResult::ATTRIBUTE_ERROR; // Invalid axis
    }

    // Prepare output tensor shape
    output->reshape(input_shape);

    // Get input and output data pointers
    const float *input_data = input.data<float>();
    size_t num_elements = input.getNumElements();

    float *output_data = new (std::nothrow) float[num_elements];
    if (!output_data)
    {
        return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR; // Memory allocation failed
    }

    // Compute strides to iterate over the axis dimension
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

    // Perform softmax computation
    for (size_t outer = 0; outer < outer_size; ++outer)
    {
        for (size_t inner = 0; inner < inner_size; ++inner)
        {
            // Compute the offset for this slice
            size_t offset = outer * axis_size * inner_size + inner;

            // Find the max value for numerical stability
            float max_val = input_data[offset];
            for (size_t i = 1; i < axis_size; ++i)
            {
                size_t idx = offset + i * inner_size;
                if (input_data[idx] > max_val)
                {
                    max_val = input_data[idx];
                }
            }

            // Compute the sum of exp(x - max)
            float sum = 0.0f;
            for (size_t i = 0; i < axis_size; ++i)
            {
                size_t idx = offset + i * inner_size;
                output_data[idx] = std::exp(input_data[idx] - max_val);
                sum += output_data[idx];
            }

            // Normalize
            for (size_t i = 0; i < axis_size; ++i)
            {
                size_t idx = offset + i * inner_size;
                output_data[idx] /= sum;
            }
        }
    }

    // Set the output data
    output->setDataType(TensorDataType::FLOAT32);
    output->setDataPointer<float>(output_data, input_shape);

    return OperatorExecuteResult::SUCCESS;
}