#ifndef ELEMENTWISE_OPERATOR_HPP
#define ELEMENTWISE_OPERATOR_HPP

#include <vector>

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
                                                      Operation op);
}

#endif // ELEMENTWISE_OPERATOR_HPP