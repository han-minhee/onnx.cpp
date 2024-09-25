#include "operator/operators.hpp"
#include <iostream>
namespace CPU_OP
{

    template <typename T>
    void transposeData(const Tensor &input_tensor, Tensor *output_tensor, const std::vector<size_t> &perm)
    {
        size_t rank = input_tensor.getNDim();
        size_t num_elements = input_tensor.getNumElements();
        const std::vector<size_t> &input_shape = input_tensor.getDims();

        // Compute output shape
        std::vector<size_t> output_shape(rank);
        for (size_t i = 0; i < rank; ++i)
        {
            output_shape[i] = input_shape[perm[i]];
        }

        // Compute strides
        const std::vector<size_t> &input_strides = input_tensor.getStrides();

        std::vector<size_t> output_strides(rank);
        output_strides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; --i)
        {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        const T *input_data = input_tensor.data<T>();
        T *output_data = new T[num_elements];

        for (size_t linear_idx = 0; linear_idx < num_elements; ++linear_idx)
        {
            size_t remaining = linear_idx;
            std::vector<size_t> idx(rank);

            for (size_t i = 0; i < rank; ++i)
            {
                idx[i] = remaining / input_strides[i];
                remaining = remaining % input_strides[i];
            }

            // Apply permutation to get output indices
            std::vector<size_t> permuted_idx(rank);
            for (size_t i = 0; i < rank; ++i)
            {
                permuted_idx[i] = idx[perm[i]];
            }

            // Compute output linear index
            size_t output_linear_idx = 0;
            for (size_t i = 0; i < rank; ++i)
            {
                output_linear_idx += permuted_idx[i] * output_strides[i];
            }

            output_data[output_linear_idx] = input_data[linear_idx];
        }

        output_tensor->setDataType(input_tensor.getDataType());
        output_tensor->setDataPointer<T>(output_data, output_shape);
    }

    OperatorExecuteResult TransposeOperatorImpl::execute(
        const std::vector<Tensor> &inputs,
        std::vector<Tensor *> &outputs,
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

        const std::vector<size_t> &input_shape = input_tensor.getDims();
        size_t rank = input_shape.size();

        // Get the 'perm' attribute, default is reversing the axes order
        std::vector<size_t> perm(rank);
        if (attributes.find("perm") != attributes.end())
        {
            const std::vector<int64_t> &perm_attr = std::get<std::vector<int64_t>>(attributes.at("perm"));

            // Ensure the length of perm matches the rank of the input tensor
            if (perm_attr.size() != rank)
            {
                return OperatorExecuteResult::ATTRIBUTE_ERROR;
            }

            // Convert perm_attr to perm
            for (size_t i = 0; i < rank; ++i)
            {
                perm[i] = static_cast<size_t>(perm_attr[i]);
            }
        }
        else
        {
            // Default permutation: reverse the dimensions
            for (size_t i = 0; i < rank; ++i)
            {
                perm[i] = rank - 1 - i;
            }
        }

        // Perform the transpose operation based on data type
        TensorDataType dtype = input_tensor.getDataType();

        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            transposeData<float>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::FLOAT64:
            transposeData<double>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::INT32:
            transposeData<int32_t>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::INT64:
            transposeData<int64_t>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::INT8:
            transposeData<int8_t>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::UINT8:
            transposeData<uint8_t>(input_tensor, output_tensor, perm);
            break;
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        return OperatorExecuteResult::SUCCESS;
    }

}