#include "operator/operators.hpp"
#include "operator/elementwise_operator.hpp"

/// FIXME: defining the lambda function inside the opertor doesn't work, but is it okay to define the lambda function here?
auto half_add = [](half_t a, half_t b)
{ return a + b; };

namespace CPU_OP
{
    OperatorExecuteResult AddOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        /// FIXME: This can be optimized by removing strides computation
        const TensorDataType dataType = inputs[0].getDataType();
        Tensor *output = outputs[0];
        std::vector<size_t> output_shape = output->getDims();
        size_t num_dims = output->getNDim();
        std::vector<size_t> output_strides = output->getStrides();
        std::vector<std::vector<size_t>> input_strides(inputs.size());
        for (size_t idx = 0; idx < inputs.size(); ++idx)
        {
            input_strides[idx] = compute_broadcast_strides(inputs[idx].getDims(), output_shape);
        }

        switch (dataType)
        {
        case TensorDataType::FLOAT32:
            return executeElementwiseOperation<float>(inputs, output, input_strides, output_strides, output_shape,
                                                      std::plus<float>());
        case TensorDataType::FLOAT64:
            return executeElementwiseOperation<double>(inputs, output, input_strides, output_strides, output_shape,
                                                       std::plus<double>());
        case TensorDataType::INT32:
            return executeElementwiseOperation<int32_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::plus<int32_t>());
        case TensorDataType::INT64:
            return executeElementwiseOperation<int64_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::plus<int64_t>());
        case TensorDataType::INT8:
            return executeElementwiseOperation<int8_t>(inputs, output, input_strides, output_strides, output_shape,
                                                       std::plus<int8_t>());
        case TensorDataType::UINT8:
            return executeElementwiseOperation<uint8_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::plus<uint8_t>());

        // for custom data types, give the operation as a lambda function
        case TensorDataType::FLOAT16:
            return executeElementwiseOperation<half_t>(inputs, output, input_strides, output_strides, output_shape, half_add);
        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }

};