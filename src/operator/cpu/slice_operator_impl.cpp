#include "operator/operators.hpp"
#include <iostream>
#include <limits>
#include <functional>

namespace CPU_OP
{
    OperatorExecuteResult SliceOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        // Validate the number of inputs
        if (inputs.size() < 3 || inputs.size() > 5)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        // Validate the output tensor
        if (outputs.empty() || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const Tensor &input_tensor = inputs[0];
        const Tensor &starts_tensor = inputs[1];
        const Tensor &ends_tensor = inputs[2];
        Tensor *output_tensor = outputs[0];

        // Check that starts and ends tensors are 1-dimensional and of the correct data type
        if (starts_tensor.getDataType() != TensorDataType::INT64 || ends_tensor.getDataType() != TensorDataType::INT64)
        {
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
        if (starts_tensor.getNDim() != 1 || ends_tensor.getNDim() != 1)
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        // Optional inputs: axes and steps
        const Tensor *axes_tensor = nullptr;
        const Tensor *steps_tensor = nullptr;

        if (inputs.size() >= 4)
        {
            axes_tensor = &inputs[3];
            if (axes_tensor->getDataType() != TensorDataType::INT64 || axes_tensor->getNDim() != 1)
            {
                return OperatorExecuteResult::DATA_TYPE_ERROR;
            }
        }

        if (inputs.size() == 5)
        {
            steps_tensor = &inputs[4];
            if (steps_tensor->getDataType() != TensorDataType::INT64 || steps_tensor->getNDim() != 1)
            {
                return OperatorExecuteResult::DATA_TYPE_ERROR;
            }
        }

        const size_t r = input_tensor.getNDim();
        const std::vector<size_t> &dims = input_tensor.getDims();

        // Prepare starts, ends, axes, and steps vectors
        const int64_t *starts_data = starts_tensor.data<int64_t>();
        const int64_t *ends_data = ends_tensor.data<int64_t>();
        size_t num_slices = starts_tensor.getNumElements();

        if (ends_tensor.getNumElements() != num_slices)
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        std::vector<int64_t> axes(num_slices);
        if (axes_tensor)
        {
            if (axes_tensor->getNumElements() != num_slices)
            {
                return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
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
                    return OperatorExecuteResult::INPUT_TENSOR_VALUE_ERROR;
                }
                axes[i] = axis;
            }
        }
        else
        {
            // If axes are omitted, they are set to [0, ..., num_slices - 1]
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
                return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
            }
            const int64_t *steps_data = steps_tensor->data<int64_t>();
            for (size_t i = 0; i < num_slices; ++i)
            {
                if (steps_data[i] == 0)
                {
                    return OperatorExecuteResult::INPUT_TENSOR_VALUE_ERROR;
                }
                steps[i] = steps_data[i];
            }
        }

        // Initialize effective starts, ends, and steps
        std::vector<int64_t> effective_starts(r, 0);
        std::vector<int64_t> effective_ends(r);
        std::vector<int64_t> effective_steps(r, 1);

        for (size_t i = 0; i < r; ++i)
        {
            effective_ends[i] = dims[i]; // dims from input tensor
        }

        // Update effective starts, ends, and steps based on axes
        for (size_t idx = 0; idx < num_slices; ++idx)
        {
            size_t axis = axes[idx];
            int64_t dim_size = static_cast<int64_t>(dims[axis]);

            int64_t start = starts_data[idx];
            int64_t end = ends_data[idx];
            int64_t step = steps[idx];

            // Adjust negative starts and ends
            if (start < 0)
            {
                start += dim_size;
            }
            if (end < 0)
            {
                end += dim_size;
            }

            // Clamping starts and ends
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

        // Compute the output shape
        std::vector<size_t> output_shape(r);
        for (size_t i = 0; i < r; ++i)
        {
            int64_t dim_size = static_cast<int64_t>(dims[i]);
            int64_t start = effective_starts[i];
            int64_t end = effective_ends[i];
            int64_t step = effective_steps[i];

            size_t dim_output = 0;

            if (step > 0)
            {
                if (start >= end)
                {
                    dim_output = 0;
                }
                else
                {
                    dim_output = static_cast<size_t>((end - start + step - 1) / step);
                }
            }
            else
            {
                if (start <= end)
                {
                    dim_output = 0;
                }
                else
                {
                    dim_output = static_cast<size_t>((start - end - step - 1) / (-step));
                }
            }
            output_shape[i] = dim_output;
        }

        // Prepare the output tensor with the public methods from Tensor
        output_tensor->reshape(output_shape);
        output_tensor->setDataType(input_tensor.getDataType());
        switch (input_tensor.getDataType())
        {
        case TensorDataType::FLOAT32:
            output_tensor->setDataPointer(new float[output_tensor->getNumElements()], output_shape);
            break;
        case TensorDataType::FLOAT64:
            output_tensor->setDataPointer(new double[output_tensor->getNumElements()], output_shape);
            break;
        case TensorDataType::INT32:
            output_tensor->setDataPointer(new int32_t[output_tensor->getNumElements()], output_shape);
            break;
        case TensorDataType::INT64:
            output_tensor->setDataPointer(new int64_t[output_tensor->getNumElements()], output_shape);
            break;
        case TensorDataType::INT8:
            output_tensor->setDataPointer(new int8_t[output_tensor->getNumElements()], output_shape);
            break;
        case TensorDataType::UINT8:
            output_tensor->setDataPointer(new uint8_t[output_tensor->getNumElements()], output_shape);
            break;
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        // Now perform slicing
        // We'll need to iterate over the output tensor and map its indices back to the input tensor

        const auto &output_dims = output_shape;
        std::function<void(size_t, size_t, std::vector<int64_t> &, std::vector<int64_t> &)> slice_recursive;
        slice_recursive = [&](size_t dim, size_t offset, std::vector<int64_t> &input_idx, std::vector<int64_t> &output_idx)
        {
            if (dim == r)
            {
                switch (input_tensor.getDataType())
                {
                case TensorDataType::FLOAT32:
                    output_tensor->data<float>()[offset] = input_tensor.data<float>()[input_tensor.getLinearIndex(input_idx)];
                    break;
                case TensorDataType::FLOAT64:
                    output_tensor->data<double>()[offset] = input_tensor.data<double>()[input_tensor.getLinearIndex(input_idx)];
                    break;
                case TensorDataType::INT32:
                    output_tensor->data<int32_t>()[offset] = input_tensor.data<int32_t>()[input_tensor.getLinearIndex(input_idx)];
                    break;
                case TensorDataType::INT64:
                    output_tensor->data<int64_t>()[offset] = input_tensor.data<int64_t>()[input_tensor.getLinearIndex(input_idx)];
                    break;
                case TensorDataType::INT8:
                    output_tensor->data<int8_t>()[offset] = input_tensor.data<int8_t>()[input_tensor.getLinearIndex(input_idx)];
                    break;
                case TensorDataType::UINT8:
                    output_tensor->data<uint8_t>()[offset] = input_tensor.data<uint8_t>()[input_tensor.getLinearIndex(input_idx)];
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in SliceOperator");
                }
                return;
            }

            int64_t start = effective_starts[dim];
            int64_t end = effective_ends[dim];
            int64_t step = effective_steps[dim];
            size_t output_dim_size = output_dims[dim];

            for (size_t i = 0; i < output_dim_size; ++i)
            {
                input_idx[dim] = start + step * i;
                output_idx[dim] = i;
                size_t new_offset = offset * output_dim_size + i;
                slice_recursive(dim + 1, new_offset, input_idx, output_idx);
            }
        };

        std::vector<int64_t> input_idx(r, 0);
        std::vector<int64_t> output_idx(r, 0);

        try
        {
            slice_recursive(0, 0, input_idx, output_idx);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during slicing: " << e.what() << std::endl;
            return OperatorExecuteResult::UNKNOWN_ERROR;
        }

        return OperatorExecuteResult::SUCCESS;
    }

}