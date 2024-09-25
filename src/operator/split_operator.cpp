#include "operator/operators.hpp"
#include <iostream>
#include <stdexcept>

std::vector<std::vector<size_t>> SplitOperator::inferOutputShapes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{

    if (inputs.empty())
    {
        throw std::invalid_argument("1");
    }

    const Tensor &input = inputs[0];
    const std::vector<size_t> &input_shape = input.getDims();
    size_t rank = input_shape.size();

    int64_t axis = 0;
    if (attributes.count("axis"))
    {
        axis = std::get<int64_t>(attributes.at("axis"));
    }

    if (axis < 0)
    {
        axis += rank;
    }
    if (axis < 0 || static_cast<size_t>(axis) >= rank)
    {
        throw std::invalid_argument("2");
    }

    size_t dim_at_axis = input_shape[axis];

    std::vector<size_t> split_sizes;
    if (inputs.size() == 2)
    {
        const Tensor &split_tensor = inputs[1];
        if (split_tensor.getDataType() != TensorDataType::INT64)
        {
            throw std::invalid_argument("3");
        }

        const int64_t *split_data = split_tensor.data<int64_t>();
        size_t num_splits = split_tensor.getNumElements();
        split_sizes.resize(num_splits);
        size_t total_size = 0;

        for (size_t i = 0; i < num_splits; ++i)
        {
            int64_t size = split_data[i];
            if (size < 0)
            {
                throw std::invalid_argument("4");
            }
            split_sizes[i] = static_cast<size_t>(size);
            total_size += split_sizes[i];
        }

        if (total_size != dim_at_axis)
        {
            throw std::invalid_argument("5");
        }
    }
    else if (attributes.count("num_outputs"))
    {

        int64_t num_outputs = std::get<int64_t>(attributes.at("num_outputs"));
        if (num_outputs <= 0)
        {
            throw std::invalid_argument("6");
        }

        split_sizes.resize(num_outputs, dim_at_axis / num_outputs);
        size_t remainder = dim_at_axis % num_outputs;
        for (size_t i = 0; i < remainder; ++i)
        {
            split_sizes[i] += 1;
        }
    }
    else
    {
        throw std::invalid_argument("7");
    }

    std::vector<std::vector<size_t>> output_shapes(split_sizes.size(), input_shape);
    for (size_t i = 0; i < split_sizes.size(); ++i)
    {
        output_shapes[i][axis] = split_sizes[i];
    }

    return output_shapes;
}

std::vector<TensorDataType> SplitOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    int64_t num_outputs = 0;
    if (attributes.count("num_outputs"))
    {
        num_outputs = std::get<int64_t>(attributes.at("num_outputs"));
    }
    else if (inputs.size() == 2)
    {
        const Tensor &split_tensor = inputs[1];
        num_outputs = split_tensor.getNumElements();
    }
    else
    {
        throw std::invalid_argument("Either 'split' input or 'num_outputs' attribute must be specified.");
    }

    std::vector<TensorDataType> dtypes(num_outputs, inputs[0].getDataType());
    return dtypes;
}

OperatorExecuteResult SplitOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                             const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    std::vector<std::vector<size_t>> output_shapes;
    try
    {
        output_shapes = inferOutputShapes(inputs, attributes);
    }
    catch (const std::invalid_argument &e)
    {
        std::string error_code = e.what();

        if (error_code == "1")
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        else if (error_code == "2")
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        else if (error_code == "3")
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        else if (error_code == "4")
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        else if (error_code == "5")
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        else if (error_code == "6")
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        else if (error_code == "7")
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        else
            return OperatorExecuteResult::UNKNOWN_ERROR;
    }

    if (outputs.size() != output_shapes.size())
    {
        return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
    }

    const Tensor &input = inputs.at(0);

    int64_t axis = 0;
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

    std::vector<size_t> split_sizes(output_shapes.size());
    for (size_t i = 0; i < output_shapes.size(); ++i)
    {
        split_sizes[i] = output_shapes[i][axis];
    }

    TensorDataType data_type = input.getDataType();

    if (data_type != TensorDataType::FLOAT32)
    {
        return OperatorExecuteResult::UNSUPPORTED_OPERATION;
    }

    const float *input_data = input.data<float>();

    size_t outer_dim = 1;
    for (size_t dim = 0; dim < static_cast<size_t>(axis); ++dim)
    {
        outer_dim *= input_shape[dim];
    }

    size_t inner_dim = 1;
    for (size_t dim = static_cast<size_t>(axis) + 1; dim < rank; ++dim)
    {
        inner_dim *= input_shape[dim];
    }

    size_t input_axis_dim = input_shape[axis];
    size_t offset = 0;
    size_t num_splits = split_sizes.size();

    for (size_t i = 0; i < num_splits; ++i)
    {
        size_t split_size = split_sizes[i];

        Tensor *output = outputs[i];
        if (output == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const std::vector<size_t> &output_shape = output_shapes[i];

        size_t num_elements = outer_dim * split_size * inner_dim;
        float *output_data = new (std::nothrow) float[num_elements];
        if (!output_data)
        {
            return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
        }

        for (size_t outer = 0; outer < outer_dim; ++outer)
        {
            for (size_t k = 0; k < split_size; ++k)
            {
                size_t input_axis_index = offset + k;
                for (size_t inner = 0; inner < inner_dim; ++inner)
                {

                    size_t input_idx = outer * input_axis_dim * inner_dim + input_axis_index * inner_dim + inner;

                    size_t output_idx = outer * split_size * inner_dim + k * inner_dim + inner;
                    output_data[output_idx] = input_data[input_idx];
                }
            }
        }

        output->setDataType(data_type);
        output->setDataPointer<float>(output_data, output_shape);

        offset += split_size;
    }

    return OperatorExecuteResult::SUCCESS;
}