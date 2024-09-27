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
    void debugPrint(const void *device_ptr, size_t num_elements, const std::string &name)
    {
        std::vector<T> host_data(num_elements);
        hipMemcpy(host_data.data(), device_ptr, num_elements * sizeof(T), hipMemcpyDeviceToHost);

        std::cout << "First 5 elements of " << name << ": ";
        for (size_t i = 0; i < std::min(num_elements, size_t(5)); ++i)
        {
            std::cout << host_data[i] << " ";
        }
        std::cout << std::endl;

        // last 5 elements
        std::cout << "Last 5 elements of " << name << ": ";
        for (size_t i = std::max(num_elements, size_t(5)) - 5; i < num_elements; ++i)
        {
            std::cout << host_data[i] << " ";
        }
        std::cout << std::endl;
    }

    // HIP kernel for element-wise addition
    __global__ void addKernel(const float *A, const float *B, float *C, size_t num_elements)
    {
        size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < num_elements)
        {
            C[idx] = A[idx] + B[idx];
        }
    }

    __global__ void printAB(const float *A, const float *B, size_t num_elements)
    {
        size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < num_elements)
        {
            printf("A[%d] = %f, B[%d] = %f\n", idx, A[idx], idx, B[idx]);
        }
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
        // if (C->getDims() != output_dims)
        // {
        //     *C = Tensor(dtype, output_dims, device);
        // }

        // Get the data pointers
        const void *A_data = A.getBuffer()->getDataPointer();
        const void *B_data = B.getBuffer()->getDataPointer();
        void *C_data = C->getBuffer()->getDataPointer();

        // temporarily, let's make a C_data, and not get it from the output tensor
        // void *C_data; // this will be used to allocate hip memory
        // hipMalloc(&C_data, C->getNumElements() * sizeof(float));

        // Get the number of elements
        size_t num_elements_A = A.getNumElements();
        size_t num_elements_B = B.getNumElements();
        size_t num_elements_C = C->getNumElements();

        // Debug prints
        debugPrint<float>(A_data, num_elements_A, "Tensor A");
        debugPrint<float>(B_data, num_elements_B, "Tensor B");
        debugPrint<float>(C_data, num_elements_C, "Tensor C (before operation)");

        // Launch the kernel to perform element-wise addition
        size_t blockSize = 256;
        size_t numBlocks = (num_elements_C + blockSize - 1) / blockSize;

        hipKernelLaunchCheck(hipLaunchKernelGGL(addKernel, dim3(numBlocks), dim3(blockSize), 0, 0, (const float *)A_data, (const float *)B_data, (float *)C_data, num_elements_C));

        hipKernelLaunchCheck(hipLaunchKernelGGL(printAB, dim3(numBlocks), dim3(blockSize), 0, 0, (const float *)A_data, (const float *)B_data, num_elements_C));

        // Wait for the kernel to finish
        hipErrorCheck(hipDeviceSynchronize());

        // Debug print after operation
        debugPrint<float>(C_data, num_elements_C, "Tensor C (after operation)");

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif
