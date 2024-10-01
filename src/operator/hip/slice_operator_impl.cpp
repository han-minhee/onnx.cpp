#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 256

namespace HIP_OP
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
    // The HIP kernel for slicing operation
    template <typename T>
    __global__ void slice_kernel(const T *__restrict__ input_data, T *__restrict__ output_data,
                                 const int64_t *__restrict__ effective_starts, const int64_t *__restrict__ effective_steps,
                                 const size_t *__restrict__ input_shape, const size_t *__restrict__ output_shape,
                                 size_t total_output_elements, size_t input_rank)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_output_elements)
            return;

        // Calculate the multi-dimensional indices for the output
        size_t linear_index = idx;
        int64_t input_idx[10]; // Assuming maximum rank is 10
        for (int i = input_rank - 1; i >= 0; --i)
        {
            size_t coord = linear_index % output_shape[i];
            linear_index /= output_shape[i];
            input_idx[i] = effective_starts[i] + coord * effective_steps[i];
        }

        // Calculate the linear index for the input tensor
        size_t input_linear_idx = 0;
        size_t stride = 1;
        for (int i = input_rank - 1; i >= 0; --i)
        {
            input_linear_idx += input_idx[i] * stride;
            stride *= input_shape[i];
        }

        // Write the output data
        output_data[idx] = input_data[input_linear_idx];
    }

    // The function to execute the slicing operation on the GPU
    template <typename T>
    OperatorExecuteResult executeSlice(const Tensor &input_tensor, Tensor *output_tensor,
                                       const std::vector<int64_t> &effective_starts, const std::vector<int64_t> &effective_steps,
                                       Device *device)
    {
        const std::vector<size_t> &input_shape = input_tensor.getDims();
        const std::vector<size_t> &output_shape = output_tensor->getDims();
        size_t num_elements = output_tensor->getNumElements();
        size_t input_rank = input_shape.size();

        const void *input_data = input_tensor.getDataPointer();
        void *output_data = output_tensor->getDataPointer();

        // Allocate device memory for the effective starts, steps, and shapes
        int64_t *d_effective_starts;
        int64_t *d_effective_steps;
        size_t *d_input_shape;
        size_t *d_output_shape;

        hipErrorCheck(hipMalloc(&d_effective_starts, sizeof(int64_t) * input_rank));
        hipErrorCheck(hipMalloc(&d_effective_steps, sizeof(int64_t) * input_rank));
        hipErrorCheck(hipMalloc(&d_input_shape, sizeof(size_t) * input_rank));
        hipErrorCheck(hipMalloc(&d_output_shape, sizeof(size_t) * input_rank));

        // Copy data to device
        hipErrorCheck(hipMemcpy(d_effective_starts, effective_starts.data(), sizeof(int64_t) * input_rank, hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_effective_steps, effective_steps.data(), sizeof(int64_t) * input_rank, hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_input_shape, input_shape.data(), sizeof(size_t) * input_rank, hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_output_shape, output_shape.data(), sizeof(size_t) * input_rank, hipMemcpyHostToDevice));

        // Prepare grid and block size
        dim3 gridSize((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE);

        // Launch the slicing kernel
        hipKernelLaunchCheck(hipLaunchKernelGGL(slice_kernel<T>, gridSize, blockSize, 0, 0,
                                                static_cast<const T *>(input_data), static_cast<T *>(output_data),
                                                d_effective_starts, d_effective_steps, d_input_shape, d_output_shape,
                                                num_elements, input_rank));

        // Free device memory
        hipErrorCheck(hipFree(d_effective_starts));
        hipErrorCheck(hipFree(d_effective_steps));
        hipErrorCheck(hipFree(d_input_shape));
        hipErrorCheck(hipFree(d_output_shape));

        return OperatorExecuteResult::SUCCESS;
    }

    // The execute function that handles different data types
    OperatorExecuteResult SliceOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
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

        TensorDataType dtype = input_tensor.getDataType();
        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            return executeSlice<float>(input_tensor, output_tensor, effective_starts, effective_steps, device);
        case TensorDataType::FLOAT64:
            return executeSlice<double>(input_tensor, output_tensor, effective_starts, effective_steps, device);
        case TensorDataType::INT32:
            return executeSlice<int32_t>(input_tensor, output_tensor, effective_starts, effective_steps, device);
        case TensorDataType::INT64:
            return executeSlice<int64_t>(input_tensor, output_tensor, effective_starts, effective_steps, device);
        case TensorDataType::FLOAT16:
            return executeSlice<__half>(input_tensor, output_tensor, effective_starts, effective_steps, device);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
        return OperatorExecuteResult::SUCCESS;
    }
};
#endif
