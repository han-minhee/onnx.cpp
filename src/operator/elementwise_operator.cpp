#include "operator/elementwise_operator.hpp"

size_t computeOffset(const std::vector<size_t> &indices,
                     const std::vector<size_t> &strides,
                     const std::vector<size_t> &adjusted_shape)
{
    size_t offset = 0;
    for (size_t dim = 0; dim < indices.size(); ++dim)
    {
        size_t index = (adjusted_shape[dim] == 1) ? 0 : indices[dim];
        offset += index * strides[dim];
    }
    return offset;
}
std::vector<std::vector<size_t>> inferElementwiseOutputShapes(const std::vector<Tensor> &inputs)
{
    std::vector<std::vector<size_t>> input_shapes;
    for (const auto &tensor : inputs)
    {
        input_shapes.push_back(tensor.getDims());
    }
    return {compute_broadcast_shape(input_shapes)};
}

std::vector<TensorDataType> inferElementwiseOutputDataTypes(const std::vector<Tensor> &inputs)
{
    if (inputs.empty())
    {
        return {};
    }
    const auto dataType = inputs[0].getDataType();
    for (const auto &input : inputs)
    {
        if (input.getDataType() != dataType)
        {
            return {};
        }
    }
    return {dataType};
}

