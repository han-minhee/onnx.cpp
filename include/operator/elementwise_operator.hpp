#ifndef ELEMENTWISE_OPERATOR_HPP
#define ELEMENTWISE_OPERATOR_HPP

#include <vector>

#include "types/half_t.hpp"
#include "tensor/tensor.hpp"
#include "operator/operator.hpp"

std::vector<std::vector<size_t>> inferElementwiseOutputShapes(const std::vector<Tensor> &inputs);
std::vector<TensorDataType> inferElementwiseOutputDataTypes(const std::vector<Tensor> &inputs);

size_t computeOffset(const std::vector<size_t> &indices,
                     const std::vector<size_t> &strides,
                     const std::vector<size_t> &adjusted_shape);

namespace CPU_OP
{
    template <typename T, typename Operation>
    OperatorExecuteResult executeElementwiseOperation(const std::vector<Tensor> &inputs,
                                                      Tensor *output,
                                                      const std::vector<std::vector<size_t>> &input_strides,
                                                      const std::vector<size_t> &output_strides,
                                                      const std::vector<size_t> &output_shape,
                                                      Operation op)
    {
        const size_t num_elements = output->getNumElements();
        T *output_data = output->data<T>();
        std::vector<const T *> input_data_ptrs(inputs.size());
        std::vector<std::vector<size_t>> adjusted_shapes(inputs.size());

        for (size_t idx = 0; idx < inputs.size(); ++idx)
        {
            input_data_ptrs[idx] = inputs[idx].data<T>();
            const auto &input_shape = inputs[idx].getDims();
            size_t input_rank = input_shape.size();

            adjusted_shapes[idx] = input_shape;
            if (input_rank < output_shape.size())
            {
                adjusted_shapes[idx].insert(adjusted_shapes[idx].begin(),
                                            output_shape.size() - input_rank, 1);
            }
        }

        const size_t num_dims = output_shape.size();
        std::vector<size_t> indices(num_dims, 0);

        for (size_t idx = 0; idx < num_elements; ++idx)
        {
            size_t remainder = idx;
            for (size_t dim = 0; dim < num_dims; ++dim)
            {
                indices[dim] = remainder / output_strides[dim];
                remainder %= output_strides[dim];
            }

            T result = input_data_ptrs[0][computeOffset(indices, input_strides[0], adjusted_shapes[0])];
            for (size_t input_idx = 1; input_idx < inputs.size(); ++input_idx)
            {
                T operand = input_data_ptrs[input_idx][computeOffset(indices, input_strides[input_idx], adjusted_shapes[input_idx])];
                result = op(result, operand);
            }
            output_data[idx] = result;
        }

        return OperatorExecuteResult::SUCCESS;
    }
}

#endif // ELEMENTWISE_OPERATOR_HPP
