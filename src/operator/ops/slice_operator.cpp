#include "operator/operators.hpp"
#include <iostream>
#include <limits>
#include <functional>

std::vector<std::vector<size_t>> SliceOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                  const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    // Validate the number of inputs
    if (inputs.size() < 3 || inputs.size() > 5)
    {
        throw std::invalid_argument("Invalid number of input tensors for SliceOperator.");
    }

    const Tensor &input_tensor = inputs[0];
    const Tensor &starts_tensor = inputs[1];
    const Tensor &ends_tensor = inputs[2];

    // Check that starts and ends tensors are 1-dimensional and of the correct data type
    if (starts_tensor.getDataType() != TensorDataType::INT64 || ends_tensor.getDataType() != TensorDataType::INT64)
    {
        throw std::invalid_argument("Starts and ends tensors must be of INT64 data type.");
    }
    if (starts_tensor.getNDim() != 1 || ends_tensor.getNDim() != 1)
    {
        throw std::invalid_argument("Starts and ends tensors must be 1-dimensional.");
    }

    // Optional inputs: axes and steps
    const Tensor *axes_tensor = nullptr;
    const Tensor *steps_tensor = nullptr;

    if (inputs.size() >= 4)
    {
        axes_tensor = &inputs[3];
        if (axes_tensor->getDataType() != TensorDataType::INT64 || axes_tensor->getNDim() != 1)
        {
            throw std::invalid_argument("Axes tensor must be of INT64 data type and 1-dimensional.");
        }
    }

    if (inputs.size() == 5)
    {
        steps_tensor = &inputs[4];
        if (steps_tensor->getDataType() != TensorDataType::INT64 || steps_tensor->getNDim() != 1)
        {
            throw std::invalid_argument("Steps tensor must be of INT64 data type and 1-dimensional.");
        }
    }

    const size_t r = input_tensor.getNDim();
    const std::vector<size_t> &dims = input_tensor.getDims();

    const int64_t *starts_data = starts_tensor.data<int64_t>();
    const int64_t *ends_data = ends_tensor.data<int64_t>();
    size_t num_slices = starts_tensor.getNumElements();

    if (ends_tensor.getNumElements() != num_slices)
    {
        throw std::invalid_argument("Mismatch in the number of elements between starts and ends tensors.");
    }

    std::vector<int64_t> axes(num_slices);
    if (axes_tensor)
    {
        if (axes_tensor->getNumElements() != num_slices)
        {
            throw std::invalid_argument("Mismatch in the number of elements between axes and other tensors.");
        }
        const int64_t *axes_data = axes_tensor->data<int64_t>();
        for (size_t i = 0; i < num_slices; ++i)
        {
            int64_t axis = axes_data[i];
            if (axis < 0)
            {
                axis += r;
            }
            if (axis < 0 || axis >= static_cast<int64_t>(r))
            {
                throw std::out_of_range("Axis value out of range.");
            }
            axes[i] = axis;
        }
    }
    else
    {
        for (size_t i = 0; i < num_slices; ++i)
        {
            axes[i] = i;
        }
    }

    std::vector<int64_t> steps(num_slices, 1);
    if (steps_tensor)
    {
        if (steps_tensor->getNumElements() != num_slices)
        {
            throw std::invalid_argument("Mismatch in the number of elements between steps and other tensors.");
        }
        const int64_t *steps_data = steps_tensor->data<int64_t>();
        for (size_t i = 0; i < num_slices; ++i)
        {
            if (steps_data[i] == 0)
            {
                throw std::invalid_argument("Step value cannot be zero.");
            }
            steps[i] = steps_data[i];
        }
    }

    std::vector<int64_t> effective_starts(r, 0);
    std::vector<int64_t> effective_ends(r);
    std::vector<int64_t> effective_steps(r, 1);

    for (size_t i = 0; i < r; ++i)
    {
        effective_ends[i] = dims[i];
    }

    for (size_t idx = 0; idx < num_slices; ++idx)
    {
        size_t axis = axes[idx];
        int64_t dim_size = static_cast<int64_t>(dims[axis]);

        int64_t start = starts_data[idx];
        int64_t end = ends_data[idx];
        int64_t step = steps[idx];

        if (start < 0)
        {
            start += dim_size;
        }
        if (end < 0)
        {
            end += dim_size;
        }

        if (step > 0)
        {
            start = std::max(int64_t(0), std::min(start, dim_size));
            end = std::max(int64_t(0), std::min(end, dim_size));
        }
        else
        {
            start = std::max(int64_t(-1), std::min(start, dim_size - 1));
            end = std::max(int64_t(-1), std::min(end, dim_size - 1));
        }

        effective_starts[axis] = start;
        effective_ends[axis] = end;
        effective_steps[axis] = step;
    }

    std::vector<size_t> output_shape(r);
    for (size_t i = 0; i < r; ++i)
    {
        int64_t start = effective_starts[i];
        int64_t end = effective_ends[i];
        int64_t step = effective_steps[i];

        size_t dim_output = 0;
        if (step > 0)
        {
            if (start < end)
            {
                dim_output = static_cast<size_t>((end - start + step - 1) / step);
            }
        }
        else
        {
            if (start > end)
            {
                dim_output = static_cast<size_t>((start - end - step - 1) / (-step));
            }
        }
        output_shape[i] = dim_output;
    }

    return {output_shape};
}

std::vector<TensorDataType> SliceOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.empty())
    {
        throw std::invalid_argument("No input tensors provided.");
    }

    return {inputs[0].getDataType()};
}

OperatorExecuteResult SliceOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                             const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device &device)
{
    DeviceType deviceType = device.getType();
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::SliceOperatorImpl::execute(inputs, outputs, attributes);

#ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::SliceOperatorImpl::execute(inputs, outputs, attributes, device);
#endif

    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
