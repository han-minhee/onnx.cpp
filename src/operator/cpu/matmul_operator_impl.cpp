
#include "operator/operators.hpp"
#include <numeric>

namespace CPU_OP
{
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

        TensorDataType dtype_A = A.getDataType();
        TensorDataType dtype_B = B.getDataType();

        if (dtype_A != dtype_B)
        {
            return OperatorExecuteResult::DATA_TYPE_ERROR;
        }

        if (dtype_A != TensorDataType::FLOAT32 && dtype_A != TensorDataType::INT32)
        {
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }

        const std::vector<size_t> &shape_A = A.getDims();
        const std::vector<size_t> &shape_B = B.getDims();

        std::vector<size_t> batch_dims_A(shape_A.begin(), shape_A.end() - 2);
        std::vector<size_t> batch_dims_B(shape_B.begin(), shape_B.end() - 2);

        std::vector<size_t> output_batch_dims;
        size_t max_rank = std::max(batch_dims_A.size(), batch_dims_B.size());

        batch_dims_A.insert(batch_dims_A.begin(), max_rank - batch_dims_A.size(), 1);
        batch_dims_B.insert(batch_dims_B.begin(), max_rank - batch_dims_B.size(), 1);

        for (size_t i = 0; i < max_rank; ++i)
        {
            size_t dim_A = batch_dims_A[i];
            size_t dim_B = batch_dims_B[i];

            if (dim_A == dim_B || dim_A == 1 || dim_B == 1)
            {
                output_batch_dims.push_back(std::max(dim_A, dim_B));
            }
            else
            {
                return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
            }
        }

        size_t dim_A_row = shape_A[shape_A.size() - 2];
        size_t dim_A_col = shape_A[shape_A.size() - 1];

        size_t dim_B_row = shape_B[shape_B.size() - 2];
        size_t dim_B_col = shape_B[shape_B.size() - 1];

        if (dim_A_col != dim_B_row)
        {
            return OperatorExecuteResult::SHAPE_MISMATCH_ERROR;
        }

        std::vector<size_t> output_shape = output_batch_dims;
        output_shape.push_back(dim_A_row);
        output_shape.push_back(dim_B_col);

        size_t num_elements_Y = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());

        if (dtype_A == TensorDataType::FLOAT32)
        {
            const float *A_data = A.data<float>();
            const float *B_data = B.data<float>();
            float *Y_data = new (std::nothrow) float[num_elements_Y];
            if (!Y_data)
            {
                return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
            }

            if (shape_A.size() == 2 && shape_B.size() == 2)
            {
                size_t M = dim_A_row;
                size_t K = dim_A_col;
                size_t N = dim_B_col;

                for (size_t i = 0; i < M; ++i)
                {
                    for (size_t j = 0; j < N; ++j)
                    {
                        float sum = 0.0f;
                        for (size_t k = 0; k < K; ++k)
                        {
                            sum += A_data[i * K + k] * B_data[k * N + j];
                        }
                        Y_data[i * N + j] = sum;
                    }
                }
                Y->setDataPointer<float>(Y_data, {M, N});
            }
            else
            {
                delete[] Y_data;
                return OperatorExecuteResult::UNSUPPORTED_OPERATION;
            }
        }
        else if (dtype_A == TensorDataType::INT32)
        {
            const int32_t *A_data = A.data<int32_t>();
            const int32_t *B_data = B.data<int32_t>();
            int32_t *Y_data = new (std::nothrow) int32_t[num_elements_Y];
            if (!Y_data)
            {
                return OperatorExecuteResult::MEMORY_ALLOCATION_ERROR;
            }

            if (shape_A.size() == 2 && shape_B.size() == 2)
            {
                size_t M = dim_A_row;
                size_t K = dim_A_col;
                size_t N = dim_B_col;

                for (size_t i = 0; i < M; ++i)
                {
                    for (size_t j = 0; j < N; ++j)
                    {
                        int32_t sum = 0;
                        for (size_t k = 0; k < K; ++k)
                        {
                            sum += A_data[i * K + k] * B_data[k * N + j];
                        }
                        Y_data[i * N + j] = sum;
                    }
                }
                Y->setDataPointer<int32_t>(Y_data, {M, N});
            }
            else
            {
                delete[] Y_data;
                return OperatorExecuteResult::UNSUPPORTED_OPERATION;
            }
        }
        else
        {
            return OperatorExecuteResult::UNSUPPORTED_OPERATION;
        }

        return OperatorExecuteResult::SUCCESS;
    }

}