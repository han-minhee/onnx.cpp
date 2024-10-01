#include "operator/operators.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>

namespace CPU_OP
{
    template <typename T>
    void executeTranspose(const Tensor &input_tensor, Tensor *output_tensor, const std::vector<size_t> &perm)
    {
        size_t rank = input_tensor.getNDim();
        size_t num_elements = input_tensor.getNumElements();
        const std::vector<size_t> &input_shape = input_tensor.getDims();

        const T *input_data = input_tensor.data<T>();
        T *output_data = output_tensor->data<T>();

        std::vector<size_t> output_strides = output_tensor->getStrides();

        for (size_t linear_idx = 0; linear_idx < num_elements; ++linear_idx)
        {
            size_t remaining = linear_idx, output_linear_idx = 0;
            for (size_t i = 0; i < rank; ++i)
            {
                size_t idx = remaining / input_tensor.getStrides()[i];
                remaining %= input_tensor.getStrides()[i];
                output_linear_idx += idx * output_strides[perm[i]];
            }
            output_data[output_linear_idx] = input_data[linear_idx];
        }
    }

    OperatorExecuteResult TransposeOperatorImpl::execute(
        const std::vector<Tensor> &inputs,
        std::vector<Tensor *> &outputs,
        const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        if (inputs.size() != 1)
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;

        if (outputs.empty() || outputs[0] == nullptr)
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;

        const Tensor &input_tensor = inputs[0];
        Tensor *output_tensor = outputs[0];

        const std::vector<size_t> &input_shape = input_tensor.getDims();
        size_t rank = input_shape.size();

        std::vector<size_t> perm(rank);
        if (attributes.count("perm"))
        {
            const auto &perm_attr = std::get<std::vector<int64_t>>(attributes.at("perm"));
            if (perm_attr.size() != rank)
                return OperatorExecuteResult::ATTRIBUTE_ERROR;
            std::transform(perm_attr.begin(), perm_attr.end(), perm.begin(), [](int64_t val)
                           { return static_cast<size_t>(val); });
        }
        else
        {
            std::iota(perm.rbegin(), perm.rend(), 0); // Default reverse permutation
        }

        // Perform the transpose operation based on data type
        switch (input_tensor.getDataType())
        {
        case TensorDataType::FLOAT32:
            executeTranspose<float>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::FLOAT64:
            executeTranspose<double>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::INT32:
            executeTranspose<int32_t>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::INT64:
            executeTranspose<int64_t>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::INT8:
            executeTranspose<int8_t>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::UINT8:
            executeTranspose<uint8_t>(input_tensor, output_tensor, perm);
            break;
        case TensorDataType::FLOAT16:
            executeTranspose<half_t>(input_tensor, output_tensor, perm);
            break;
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        return OperatorExecuteResult::SUCCESS;
    }
}
