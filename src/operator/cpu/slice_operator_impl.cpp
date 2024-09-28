#include "operator/operators.hpp"
#include <iostream>
#include <limits>
#include <functional>

namespace CPU_OP
{
    std::vector<int64_t> prepareAxes(const Tensor *axes_tensor, size_t num_slices, size_t r)
    {
        std::vector<int64_t> axes(num_slices);

        if (axes_tensor)
        {
            const int64_t *axes_data = axes_tensor->data<int64_t>();
            for (size_t i = 0; i < num_slices; ++i)
            {
                int64_t axis = axes_data[i];
                if (axis < 0)
                    axis += r;
                if (axis < 0 || axis >= static_cast<int64_t>(r))
                    throw std::runtime_error("Axis out of bounds in SliceOperator");
                axes[i] = axis;
            }
        }
        else
        {
            for (size_t i = 0; i < num_slices; ++i)
                axes[i] = i;
        }
        return axes;
    }

    std::vector<int64_t> prepareSteps(const Tensor *steps_tensor, size_t num_slices)
    {
        std::vector<int64_t> steps(num_slices, 1);

        if (steps_tensor)
        {
            const int64_t *steps_data = steps_tensor->data<int64_t>();
            for (size_t i = 0; i < num_slices; ++i)
            {
                if (steps_data[i] == 0)
                    throw std::runtime_error("Step value cannot be zero in SliceOperator");
                steps[i] = steps_data[i];
            }
        }
        return steps;
    }

    void computeEffectiveStartEndSteps(const Tensor &input_tensor, const std::vector<int64_t> &starts_data,
                                       const std::vector<int64_t> &ends_data, const std::vector<int64_t> &axes,
                                       const std::vector<int64_t> &steps, std::vector<int64_t> &effective_starts,
                                       std::vector<int64_t> &effective_ends, std::vector<int64_t> &effective_steps)
    {
        const auto &dims = input_tensor.getDims();
        size_t r = dims.size();

        for (size_t i = 0; i < r; ++i)
            effective_ends[i] = dims[i];

        for (size_t idx = 0; idx < axes.size(); ++idx)
        {
            size_t axis = axes[idx];
            int64_t dim_size = static_cast<int64_t>(dims[axis]);

            int64_t start = starts_data[idx], end = ends_data[idx], step = steps[idx];
            if (start < 0)
                start += dim_size;
            if (end < 0)
                end += dim_size;

            if (step > 0)
            {
                start = std::clamp(start, int64_t(0), dim_size);
                end = std::clamp(end, int64_t(0), dim_size);
            }
            else
            {
                start = std::clamp(start, int64_t(-1), dim_size - 1);
                end = std::clamp(end, int64_t(-1), dim_size - 1);
            }

            effective_starts[axis] = start;
            effective_ends[axis] = end;
            effective_steps[axis] = step;
        }
    }


    template <typename T>
    void performSlicingRecursive(const Tensor &input_tensor, Tensor *output_tensor, size_t dim, size_t offset,
                                 const std::vector<int64_t> &effective_starts, const std::vector<int64_t> &effective_steps,
                                 const std::vector<size_t> &output_dims, std::vector<int64_t> &input_idx, std::vector<int64_t> &output_idx)
    {
        size_t r = input_tensor.getNDim();

        if (dim == r)
        {
            size_t input_index = input_tensor.getLinearIndex(input_idx);
            size_t output_index = offset;

            const T *input_data = static_cast<const T *>(input_tensor.getDataPointer());
            T *output_data = static_cast<T *>(output_tensor->getDataPointer());

            output_data[output_index] = input_data[input_index];
            return;
        }

        int64_t start = effective_starts[dim], step = effective_steps[dim];
        size_t output_dim_size = output_dims[dim];

        for (size_t i = 0; i < output_dim_size; ++i)
        {
            input_idx[dim] = start + step * i;
            output_idx[dim] = i;
            size_t new_offset = offset * output_dim_size + i;
            performSlicingRecursive<T>(input_tensor, output_tensor, dim + 1, new_offset, effective_starts, effective_steps, output_dims, input_idx, output_idx);
        }
    }

    void performSlicing(const Tensor &input_tensor, Tensor *output_tensor, size_t dim, size_t offset,
                        const std::vector<int64_t> &effective_starts, const std::vector<int64_t> &effective_steps,
                        const std::vector<size_t> &output_dims, std::vector<int64_t> &input_idx, std::vector<int64_t> &output_idx)
    {
        switch (input_tensor.getDataType())
        {
        case TensorDataType::FLOAT32:
            performSlicingRecursive<float>(input_tensor, output_tensor, dim, offset, effective_starts, effective_steps, output_dims, input_idx, output_idx);
            break;
        case TensorDataType::FLOAT64:
            performSlicingRecursive<double>(input_tensor, output_tensor, dim, offset, effective_starts, effective_steps, output_dims, input_idx, output_idx);
            break;
        case TensorDataType::INT32:
            performSlicingRecursive<int32_t>(input_tensor, output_tensor, dim, offset, effective_starts, effective_steps, output_dims, input_idx, output_idx);
            break;
        case TensorDataType::INT64:
            performSlicingRecursive<int64_t>(input_tensor, output_tensor, dim, offset, effective_starts, effective_steps, output_dims, input_idx, output_idx);
            break;
        case TensorDataType::INT8:
            performSlicingRecursive<int8_t>(input_tensor, output_tensor, dim, offset, effective_starts, effective_steps, output_dims, input_idx, output_idx);
            break;
        case TensorDataType::UINT8:
            performSlicingRecursive<uint8_t>(input_tensor, output_tensor, dim, offset, effective_starts, effective_steps, output_dims, input_idx, output_idx);
            break;
        default:
            throw std::runtime_error("Unsupported data type in SliceOperator");
        }
    }

    OperatorExecuteResult SliceOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        const Tensor &input_tensor = inputs[0];
        const Tensor &starts_tensor = inputs[1];
        const Tensor &ends_tensor = inputs[2];
        Tensor *output_tensor = outputs[0];

        const Tensor *axes_tensor = (inputs.size() >= 4) ? &inputs[3] : nullptr;
        const Tensor *steps_tensor = (inputs.size() == 5) ? &inputs[4] : nullptr;

        const size_t r = input_tensor.getNDim();

        std::vector<int64_t> starts_data(starts_tensor.data<int64_t>(), starts_tensor.data<int64_t>() + starts_tensor.getNumElements());
        std::vector<int64_t> ends_data(ends_tensor.data<int64_t>(), ends_tensor.data<int64_t>() + ends_tensor.getNumElements());

        std::vector<int64_t> axes = prepareAxes(axes_tensor, starts_data.size(), r);
        std::vector<int64_t> steps = prepareSteps(steps_tensor, starts_data.size());

        std::vector<int64_t> effective_starts(r, 0), effective_ends(r), effective_steps(r, 1);
        computeEffectiveStartEndSteps(input_tensor, starts_data, ends_data, axes, steps, effective_starts, effective_ends, effective_steps);
        std::vector<size_t> output_shape = output_tensor->getDims();
        std::vector<int64_t> input_idx(r, 0), output_idx(r, 0);

        try
        {
            performSlicing(input_tensor, output_tensor, 0, 0, effective_starts, effective_steps, output_shape, input_idx, output_idx);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during slicing: " << e.what() << std::endl;
            return OperatorExecuteResult::UNKNOWN_ERROR;
        }

        return OperatorExecuteResult::SUCCESS;
    }
}
