#include "operator/operators.hpp"
#include <numeric>
#include <iostream>

namespace CPU_OP
{
    template <typename T>
    OperatorExecuteResult executeMatMul(const Tensor &A, const Tensor &B, Tensor *Y,
                                        size_t dim_A_row, size_t dim_A_col, size_t dim_B_col)
    {
        const T *A_data = A.data<T>();
        const T *B_data = B.data<T>();

        if (!A_data || !B_data)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        // Allocate buffer for output data if not already allocated or if dimensions mismatch
        if (!Y->data<T>() || Y->getNumElements() != dim_A_row * dim_B_col)
        {
            Y->allocateBuffer(A.getDataType(), dim_A_row * dim_B_col);
            Y->reshape({dim_A_row, dim_B_col});
        }

        T *Y_data = Y->data<T>();
        if (!Y_data)
        {
            return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
        }

        // Initialize output data
        std::fill(Y_data, Y_data + (dim_A_row * dim_B_col), static_cast<T>(0));

        // Perform matrix multiplication
        for (size_t i = 0; i < dim_A_row; ++i)
        {
            for (size_t j = 0; j < dim_B_col; ++j)
            {
                T sum = 0;
                for (size_t k = 0; k < dim_A_col; ++k)
                {
                    sum += A_data[i * dim_A_col + k] * B_data[k * dim_B_col + j];
                }
                Y_data[i * dim_B_col + j] = sum;
            }
        }

        return OperatorExecuteResult::SUCCESS;
    }

    OperatorExecuteResult MatMulOperatorImpl::execute(const std::vector<Tensor> &inputs,
                                                      std::vector<Tensor *> &outputs,
                                                      const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        if (inputs.size() != 2)
        {
            return OperatorExecuteResult::INPUT_TENSOR_ERROR;
        }

        if (outputs.size() != 1 || outputs[0] == nullptr)
        {
            return OperatorExecuteResult::OUTPUT_TENSOR_ERROR;
        }

        const Tensor &A = inputs[0];
        const Tensor &B = inputs[1];
        Tensor *Y = outputs[0];

        const std::vector<size_t> &shape_A = A.getDims();
        const std::vector<size_t> &shape_B = B.getDims();

        if (shape_A.size() != 2 || shape_B.size() != 2)
        {
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }

        size_t dim_A_row = shape_A[0];
        size_t dim_A_col = shape_A[1];
        size_t dim_B_row = shape_B[0];
        size_t dim_B_col = shape_B[1];

        if (dim_A_col != dim_B_row)
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        // Reshape output tensor
        std::vector<size_t> output_shape = {dim_A_row, dim_B_col};
        Y->reshape(output_shape);
        Y->setDataType(A.getDataType());

        // Use appropriate data type for execution
        switch (A.getDataType())
        {
        case TensorDataType::FLOAT32:
            return executeMatMul<float>(A, B, Y, dim_A_row, dim_A_col, dim_B_col);
        case TensorDataType::FLOAT64:
            return executeMatMul<double>(A, B, Y, dim_A_row, dim_A_col, dim_B_col);
        case TensorDataType::INT32:
            return executeMatMul<int32_t>(A, B, Y, dim_A_row, dim_A_col, dim_B_col);
        case TensorDataType::INT64:
            return executeMatMul<int64_t>(A, B, Y, dim_A_row, dim_A_col, dim_B_col);
        case TensorDataType::INT8:
            return executeMatMul<int8_t>(A, B, Y, dim_A_row, dim_A_col, dim_B_col);
        case TensorDataType::UINT8:
            return executeMatMul<uint8_t>(A, B, Y, dim_A_row, dim_A_col, dim_B_col);
        default:
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }
    }
}
