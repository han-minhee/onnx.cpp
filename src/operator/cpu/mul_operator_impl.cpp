#include <functional>
#include "operator/operators.hpp"
#include "operator/elementwise_operator.hpp"

namespace CPU_OP
{

    OperatorExecuteResult MulOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {

        Tensor *output = outputs[0];
        const TensorDataType dataType = inputs[0].getDataType();
        std::vector<size_t> output_shape = output->getDims();
        
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
                                                      std::multiplies<float>());
        case TensorDataType::FLOAT64:
            return executeElementwiseOperation<double>(inputs, output, input_strides, output_strides, output_shape,
                                                       std::multiplies<double>());
        case TensorDataType::INT32:
            return executeElementwiseOperation<int32_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::multiplies<int32_t>());
        case TensorDataType::INT64:
            return executeElementwiseOperation<int64_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::multiplies<int64_t>());
        case TensorDataType::INT8:
            return executeElementwiseOperation<int8_t>(inputs, output, input_strides, output_strides, output_shape,
                                                       std::multiplies<int8_t>());
        case TensorDataType::UINT8:
            return executeElementwiseOperation<uint8_t>(inputs, output, input_strides, output_strides, output_shape,
                                                        std::multiplies<uint8_t>());
        default:
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }
    }

}