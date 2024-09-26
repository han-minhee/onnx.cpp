#include "operator/operators.hpp"
#include <iostream>

namespace CPU_OP
{

    OperatorExecuteResult ReshapeOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        // Validate the number of inputs
        if (inputs.size() != 2)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        // Validate the output tensor
        if (outputs.empty() || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const Tensor &input_tensor = inputs[0];
        const Tensor &shape_tensor = inputs[1];
        Tensor *output_tensor = outputs[0];

        // Check that the shape tensor is a 1-dimensional tensor of int64 type
        if (shape_tensor.getDataType() != TensorDataType::INT64 || shape_tensor.getNDim() != 1)
        {
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        const int64_t *shape_data = shape_tensor.data<int64_t>();
        size_t shape_size = shape_tensor.getNumElements();

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
                    return OperatorExecuteResult::SHAPE_MISMATCH_ERROR; // More than one -1 in shape
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
                return OperatorExecuteResult::SHAPE_MISMATCH_ERROR; // Invalid dimension value
            }
        }

        // Infer the dimension if -1 was provided
        if (minus_one_pos != -1)
        {
            if (input_num_elements % inferred_dimension != 0)
            {
                return OperatorExecuteResult::SHAPE_MISMATCH_ERROR; // Cannot infer the shape
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
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR; // Number of elements mismatch
        }

        // Allocate memory for the reshaped tensor and copy data from the input tensor
        float *output_data = new (std::nothrow) float[input_num_elements];
        if (!output_data)
        {
            return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
        }

        const float *input_data = input_tensor.data<float>();
        std::copy(input_data, input_data + input_num_elements, output_data);

        // Set the data pointer and shape of the output tensor
        output_tensor->setDataType(input_tensor.getDataType());
        output_tensor->setDataPointer<float>(output_data, output_shape);

        return OperatorExecuteResult::SUCCESS;
    }

}