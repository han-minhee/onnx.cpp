#include <functional>
#include "operator/operators.hpp"
#include "operator/elementwise_operator.hpp"

std::vector<std::vector<size_t>> MulOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return inferElementwiseOutputShapes(inputs);
}

std::vector<TensorDataType> MulOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                              const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    return inferElementwiseOutputDataTypes(inputs);
}

OperatorExecuteResult MulOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
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

    DeviceType deviceType = device->getType();

    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::MulOperatorImpl::execute(inputs, outputs, attributes);
#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::MulOperatorImpl::execute(inputs, outputs, attributes, device);
#endif
    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
