#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#include "utils.hpp"

#define MAX_DIMS 8

namespace HIP_OP
{

    // Helper function to compute the broadcasted shape
    std::vector<size_t> broadcastShapes(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2)
    {
        size_t ndim1 = shape1.size();
        size_t ndim2 = shape2.size();
        size_t ndim = std::max(ndim1, ndim2);

        std::vector<size_t> result_shape(ndim, 1);

        for (size_t i = 0; i < ndim; ++i)
        {
            size_t dim1 = (i < ndim - ndim1) ? 1 : shape1[i - (ndim - ndim1)];
            size_t dim2 = (i < ndim - ndim2) ? 1 : shape2[i - (ndim - ndim2)];

            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
            {
                throw std::runtime_error("Shapes are not broadcastable");
            }

            result_shape[i] = std::max(dim1, dim2);
        }

        return result_shape;
    }

    template <typename T>
    __global__ void hip_add_kernel(const T *A_data, const size_t *A_dims, const size_t *A_strides,
                                   const T *B_data, const size_t *B_dims, const size_t *B_strides,
                                   T *C_data, const size_t *C_dims, const size_t *C_strides,
                                   size_t num_elements)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;

        // Compute the multi-dimensional index for C
        size_t indices[MAX_DIMS];
        size_t tmp = idx;
        for (int i = MAX_DIMS - 1; i >= 0; --i)
        {
            indices[i] = tmp % C_dims[i];
            tmp /= C_dims[i];
        }

        // Compute the index in A
        size_t A_idx = 0;
        for (int i = 0; i < MAX_DIMS; ++i)
        {
            size_t dim = A_dims[i];
            size_t stride = A_strides[i];
            size_t index = indices[i];
            if (dim == 1)
            {
                index = 0; // broadcasting dimension
            }
            A_idx += index * stride;
        }

        // Compute the index in B
        size_t B_idx = 0;
        for (int i = 0; i < MAX_DIMS; ++i)
        {
            size_t dim = B_dims[i];
            size_t stride = B_strides[i];
            size_t index = indices[i];
            if (dim == 1)
            {
                index = 0; // broadcasting dimension
            }
            B_idx += index * stride;
        }

        // Perform the addition
        C_data[idx] = A_data[A_idx] + B_data[B_idx];
    }

    OperatorExecuteResult AddOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        // Check if the device is a HIP device
        if (device->getType() != DeviceType::HIP)
        {
            throw std::runtime_error("Device is not a HIP device");
        }

        // Check inputs and outputs size
        if (inputs.size() != 2)
        {
            throw std::runtime_error("Add operator requires exactly 2 inputs");
        }

        if (outputs.size() != 1)
        {
            throw std::runtime_error("Add operator requires exactly 1 output");
        }

        const Tensor &A = inputs[0];
        const Tensor &B = inputs[1];
        Tensor *C = outputs[0];

        // Get the data type
        TensorDataType dtype = A.getDataType();

        // Check that data types of A and B match
        if (A.getDataType() != B.getDataType())
        {
            throw std::runtime_error("Input tensors must have the same data type");
        }

        // Compute the output dimensions according to broadcasting rules
        std::vector<size_t> output_dims = broadcastShapes(A.getDims(), B.getDims());

        // Allocate the output tensor
        if (C->getDims() != output_dims)
        {
            *C = Tensor(dtype, output_dims, device);
        }

        // Get the data pointers
        const void *A_data = A.getBuffer()->getDataPointer();
        const void *B_data = B.getBuffer()->getDataPointer();
        void *C_data = C->getBuffer()->getDataPointer();

        // Prepare the dimensions and strides arrays
        size_t A_dims[MAX_DIMS], B_dims[MAX_DIMS], C_dims[MAX_DIMS];
        size_t A_strides[MAX_DIMS], B_strides[MAX_DIMS], C_strides[MAX_DIMS];

        auto fillDimsAndStrides = [](const std::vector<size_t> &dims, const std::vector<size_t> &strides,
                                     size_t *dims_array, size_t *strides_array)
        {
            size_t ndim = dims.size();
            size_t offset = MAX_DIMS - ndim;
            for (size_t i = 0; i < MAX_DIMS; ++i)
            {
                if (i < offset)
                {
                    dims_array[i] = 1;
                    strides_array[i] = 0;
                }
                else
                {
                    dims_array[i] = dims[i - offset];
                    strides_array[i] = strides[i - offset];
                }
            }
        };

        fillDimsAndStrides(A.getDims(), A.getStrides(), A_dims, A_strides);
        fillDimsAndStrides(B.getDims(), B.getStrides(), B_dims, B_strides);
        fillDimsAndStrides(C->getDims(), C->getStrides(), C_dims, C_strides);

        // Allocate device memory for dimensions and strides arrays
        size_t *d_A_dims, *d_B_dims, *d_C_dims;
        size_t *d_A_strides, *d_B_strides, *d_C_strides;

        hipErrorCheck(hipMalloc(&d_A_dims, MAX_DIMS * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_B_dims, MAX_DIMS * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_C_dims, MAX_DIMS * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_A_strides, MAX_DIMS * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_B_strides, MAX_DIMS * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_C_strides, MAX_DIMS * sizeof(size_t)));

        // Copy dimensions and strides to device memory
        hipErrorCheck(hipMemcpy(d_A_dims, A_dims, MAX_DIMS * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_B_dims, B_dims, MAX_DIMS * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_C_dims, C_dims, MAX_DIMS * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_A_strides, A_strides, MAX_DIMS * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_B_strides, B_strides, MAX_DIMS * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_C_strides, C_strides, MAX_DIMS * sizeof(size_t), hipMemcpyHostToDevice));

        // Launch the HIP kernel
        size_t num_elements = C->getNumElements();
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

        // Dispatch the kernel based on data type
        if (dtype == TensorDataType::FLOAT32)
        {
            hipLaunchKernelGGL(hip_add_kernel<float>, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                                             static_cast<const float *>(A_data), d_A_dims, d_A_strides,
                                             static_cast<const float *>(B_data), d_B_dims, d_B_strides,
                                             static_cast<float *>(C_data), d_C_dims, d_C_strides,
                                             num_elements);
        }
        else
        {
            // Free device memory
            hipErrorCheck(hipFree(d_A_dims));
            hipErrorCheck(hipFree(d_B_dims));
            hipErrorCheck(hipFree(d_C_dims));
            hipErrorCheck(hipFree(d_A_strides));
            hipErrorCheck(hipFree(d_B_strides));
            hipErrorCheck(hipFree(d_C_strides));
            throw std::runtime_error("Unsupported data type in Add operator");
        }

        // Synchronize
        hipErrorCheck(hipDeviceSynchronize());

        // Free device memory
        hipErrorCheck(hipFree(d_A_dims));
        hipErrorCheck(hipFree(d_B_dims));
        hipErrorCheck(hipFree(d_C_dims));
        hipErrorCheck(hipFree(d_A_strides));
        hipErrorCheck(hipFree(d_B_strides));
        hipErrorCheck(hipFree(d_C_strides));

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif
