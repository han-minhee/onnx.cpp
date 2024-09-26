#include "operator/operators.hpp"
#include "operator/elementwise_operator.hpp"

namespace CPU_OP
{
    OperatorExecuteResult SubOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        if (inputs.size() < 2)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        const TensorDataType dataType = inputs[0].getDataType();
        for (size_t i = 1; i < inputs.size(); i++)
        {
            if (inputs[i].getDataType() != dataType)
            {
                return OperatorExecuteResult::DATA_TYPE_ERROR;
            }
        }
        /// FIXME: output tensor should already have a shape and data type defined
        std::vector<std::vector<size_t>> input_shapes;
        for (const auto &tensor : inputs)
        {
            input_shapes.push_back(tensor.getDims());
        }
        std::vector<size_t> output_shape = compute_broadcast_shape(input_shapes);
        if (output_shape.empty())
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        if (outputs.empty() || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        Tensor *output = outputs[0];
        output->reshape(output_shape);
        size_t num_dims = output->getNDim();
        std::vector<size_t> output_strides(num_dims, 1);
        for (int i = num_dims - 2; i >= 0; --i)
        {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        std::vector<std::vector<size_t>> input_strides(inputs.size());
        for (size_t idx = 0; idx < inputs.size(); ++idx)
        {
            input_strides[idx] = compute_broadcast_strides(inputs[idx].getDims(), output_shape);
        }

        switch (dataType)
        {
        case TensorDataType::FLOAT32:
            return executeElementwiseOperation<float>(inputs, output, input_strides, output_strides, output_shape,
                                                      std::minus<float>());
        case TensorDataType::FLOAT64:
            return executeElementwiseOperation<double>(inputs, output, input_strides, output_strides, output_shape,
                                                       std::minus<double>());
        case TensorDataType::INT32:
            return executeElementwiseOperation<int32_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::minus<int32_t>());
        case TensorDataType::INT64:
            return executeElementwiseOperation<int64_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::minus<int64_t>());
        case TensorDataType::INT8:
            return executeElementwiseOperation<int8_t>(inputs, output, input_strides, output_strides, output_shape,
                                                       std::minus<int8_t>());
        case TensorDataType::UINT8:
            return executeElementwiseOperation<uint8_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::minus<uint8_t>());
        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }
}