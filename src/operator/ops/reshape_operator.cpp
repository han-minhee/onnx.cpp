#include "operator/operators.hpp"
#include <iostream>

std::vector<std::vector<size_t>> ReshapeOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    // Validate the number of inputs
    if (inputs.size() != 2)
    {
        return {};
    }

    const Tensor &input_tensor = inputs[0];
    const Tensor &shape_tensor = inputs[1];
    size_t shape_size = shape_tensor.getNumElements();

    // Check that the shape tensor is a 1-dimensional tensor of int64 type
    if (shape_tensor.getDataType() != TensorDataType::INT64 || shape_tensor.getNDim() != 1)
    {
        return {};
    }

    const int64_t *shape_data = shape_tensor.data<int64_t>();

    // Extract the "allowzero" attribute (default is 0)
    bool allowzero = false;
    if (attributes.find("allowzero") != attributes.end())
    {
        allowzero = static_cast<int64_t>(std::get<int64_t>(attributes.at("allowzero"))) != 0;
    }

    // Calculate the total number of elements in the input tensor
    size_t input_num_elements = input_tensor.getNumElements();

    // Determine the target shape and handle -1 and 0 cases
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
                return {}; // More than one -1 in shape
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
                output_shape[i] = input_tensor.getDims()[i]; // Copy the corresponding input dimension
            }
        }
        else if (dim > 0)
        {
            output_shape[i] = static_cast<size_t>(dim);
            inferred_dimension *= output_shape[i];
        }
        else
        {
            return {}; // Invalid dimension value
        }
    }

    // Infer the dimension if -1 was provided
    if (minus_one_pos != -1)
    {
        if (input_num_elements % inferred_dimension != 0)
        {
            return {}; // Cannot infer the shape
        }
        output_shape[minus_one_pos] = input_num_elements / inferred_dimension;
    }

    // Validate that the reshaped tensor has the same total number of elements as the input tensor
    size_t output_num_elements = 1;
    for (size_t dim : output_shape)
    {
        output_num_elements *= dim;
    }

    if (output_num_elements != input_num_elements)
    {
        return {}; // Number of elements mismatch
    }

    return {output_shape};
}

std::vector<TensorDataType> ReshapeOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                  const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() != 2)
    {
        return {};
    }

    const Tensor &input_tensor = inputs[0];
    const Tensor &shape_tensor = inputs[1];

    // Check that the shape tensor is a 1-dimensional tensor of int64 type
    if (shape_tensor.getDataType() != TensorDataType::INT64 || shape_tensor.getNDim() != 1)
    {
        return {};
    }

    return {input_tensor.getDataType()};
}

OperatorExecuteResult ReshapeOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                               const std::unordered_map<std::string, Node::AttributeValue> &attributes, DeviceType deviceType)
{
    switch (deviceType)
    {
    case (DeviceType::CPU):
        CPU_OP::ReshapeOperatorImpl::execute(inputs, outputs, attributes);

#ifdef USE_HIP
    case (DeviceType::HIP)
        HIP_OP::ReshapeOperatorImpl::execute(inputs, outputs, attributes);

#endif
        default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
