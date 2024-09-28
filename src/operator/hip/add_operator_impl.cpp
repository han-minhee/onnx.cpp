#ifdef USE_HIP
#include "operator/operators.hpp"
#include "operator/elementwise_operator.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#include "utils.hpp"

#define MAX_DIMS 8

namespace HIP_OP
{

    template <typename T>
    __global__ void add_kernel(const T *A_data, const size_t *A_dims, const size_t *A_strides,
                               const T *B_data, const size_t *B_dims, const size_t *B_strides,
                               T *C_data, const size_t *C_dims, const size_t *C_strides,
                               size_t num_elements, size_t A_ndims, size_t B_ndims, size_t C_ndims)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;

        // Compute the multi-dimensional index for C
        size_t indices[MAX_DIMS];
        size_t tmp = idx;
        for (int i = C_ndims - 1; i >= 0; --i)
        {
            indices[i] = tmp % C_dims[i];
            tmp /= C_dims[i];
        }

        // Compute the index in A
        size_t A_idx = 0;
        for (int i = 0; i < A_ndims; ++i)
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
        for (int i = 0; i < B_ndims; ++i)
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

        const std::vector<size_t> output_dims = C->getDims();

        // Get the data pointers
        const void *A_data = A.getBuffer()->getDataPointer();
        const void *B_data = B.getBuffer()->getDataPointer();
        void *C_data = C->getBuffer()->getDataPointer();

        // Get the number of elements
        size_t num_elements_A = A.getNumElements();
        size_t num_elements_B = B.getNumElements();
        size_t num_elements_C = C->getNumElements();

        // get the dims and strides
        const std::vector<size_t> A_dims = A.getDims();
        const std::vector<size_t> B_dims = B.getDims();
        const std::vector<size_t> C_dims = C->getDims();
        const std::vector<size_t> A_strides = A.getStrides();
        const std::vector<size_t> B_strides = B.getStrides();
        const std::vector<size_t> C_strides = C->getStrides();

        // make device pointers and allocate memory for dims and strides
        size_t *d_A_dims, *d_B_dims, *d_C_dims;
        size_t *d_A_strides, *d_B_strides, *d_C_strides;
        size_t A_ndims = A_dims.size();
        size_t B_ndims = B_dims.size();
        size_t C_ndims = C_dims.size();

        hipErrorCheck(hipMalloc(&d_A_dims, A_ndims * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_B_dims, B_ndims * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_C_dims, C_ndims * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_A_strides, A_ndims * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_B_strides, B_ndims * sizeof(size_t)));
        hipErrorCheck(hipMalloc(&d_C_strides, C_ndims * sizeof(size_t)));

        hipErrorCheck(hipMemcpy(d_A_dims, A_dims.data(), A_ndims * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_B_dims, B_dims.data(), B_ndims * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_C_dims, C_dims.data(), C_ndims * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_A_strides, A_strides.data(), A_ndims * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_B_strides, B_strides.data(), B_ndims * sizeof(size_t), hipMemcpyHostToDevice));
        hipErrorCheck(hipMemcpy(d_C_strides, C_strides.data(), C_ndims * sizeof(size_t), hipMemcpyHostToDevice));

        // launch the kernel
        hipKernelLaunchCheck(hipLaunchKernelGGL(add_kernel<float>, dim3((num_elements_C + 255) / 256), dim3(256), 0, 0,
                                                (float *)A_data, d_A_dims, d_A_strides,
                                                (float *)B_data, d_B_dims, d_B_strides,
                                                (float *)C_data, d_C_dims, d_C_strides,
                                                num_elements_C, A_ndims, B_ndims, C_ndims));

        // Wait for the kernel to finish
        hipErrorCheck(hipDeviceSynchronize());

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif
