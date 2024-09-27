
#include "operator/operators.hpp"
#include <numeric>

// implement inferOutputShapes and inferOutputDataTypes
std::vector<std::vector<size_t>> MatMulOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() != 2)
    {
        return {};
    }

    const std::vector<size_t> &shape_A = inputs[0].getDims();
    const std::vector<size_t> &shape_B = inputs[1].getDims();

    size_t dim_A_row = shape_A[shape_A.size() - 2];
    size_t dim_A_col = shape_A[shape_A.size() - 1];

    size_t dim_B_row = shape_B[shape_B.size() - 2];
    size_t dim_B_col = shape_B[shape_B.size() - 1];

    if (dim_A_col != dim_B_row)
    {
        return {};
    }

    std::vector<size_t> output_shape = {dim_A_row, dim_B_col};

    return {output_shape};
}

std::vector<TensorDataType> MatMulOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                 const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (inputs.size() != 2)
    {
        return {};
    }

    TensorDataType dtype_A = inputs[0].getDataType();
    TensorDataType dtype_B = inputs[1].getDataType();

    if (dtype_A != dtype_B)
    {
        return {};
    }

    if (dtype_A != TensorDataType::FLOAT32 && dtype_A != TensorDataType::INT32)
    {
        return {};
    }

    return {dtype_A};
}

OperatorExecuteResult MatMulOperator::execute(const std::vector<Tensor> &inputs,
                                              std::vector<Tensor *> &outputs,
                                              const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device& device)
{
    DeviceType deviceType = device.getType();  
    switch (deviceType)
    {
    case DeviceType::CPU:
        return CPU_OP::MatMulOperatorImpl::execute(inputs, outputs, attributes);

    #ifdef USE_HIP
    case DeviceType::HIP:
        return HIP_OP::MatMulOperatorImpl::execute(inputs, outputs, attributes, device);
    #endif

    default:
        return OperatorExecuteResult::DEVICE_UNSUPPORTED;
    }
}
