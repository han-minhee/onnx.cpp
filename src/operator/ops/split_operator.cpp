#include "operator/operators.hpp"
#include <iostream>
#include <stdexcept>

std::vector<std::vector<size_t>> SplitOperator::inferOutputShapes(
    const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{

    if (inputs.empty())
    {
        return {};
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

        return {};
    }

    size_t dim_at_axis = input_shape[axis];

    std::vector<size_t> split_sizes;
    if (inputs.size() == 2)
    {
        const Tensor &split_tensor = inputs[1];
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
            return {};
        }
    }
    else if (attributes.count("num_outputs"))
    {

        int64_t num_outputs = std::get<int64_t>(attributes.at("num_outputs"));
        if (num_outputs <= 0)
        {
            return {};
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
        return {};
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
                                             const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
{
    if (inputs.empty())
    {
        return OperatorExecuteResult::INPUT_TENSOR_ERROR;
    }

    if (outputs.empty())
    {
        return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
    }

    DeviceType deviceType = device->getType();
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::SplitOperatorImpl::execute(inputs, outputs, attributes);

#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::SplitOperatorImpl::execute(inputs, outputs, attributes, device);
#endif

    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}