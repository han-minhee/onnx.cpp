#include "operator/operators.hpp"
#include <iostream>

std::vector<std::vector<size_t>> ConstantOperator::inferOutputShapes(const std::vector<Tensor> &inputs,
                                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (attributes.count("value"))
    {
        return {std::get<Tensor>(attributes.at("value")).getDims()};
    }
    return {};
}

std::vector<TensorDataType> ConstantOperator::inferOutputDataTypes(const std::vector<Tensor> &inputs,
                                                                   const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (attributes.count("value"))
    {
        return {std::get<Tensor>(attributes.at("value")).getDataType()};
    }
    return {};
}

OperatorExecuteResult ConstantOperator::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                const std::unordered_map<std::string, Node::AttributeValue> &attributes)
{
    if (attributes.count("value"))
    {
        outputs[0]->copy_tensor(std::get<Tensor>(attributes.at("value")));
        return OperatorExecuteResult::SUCCESS;
    }
    /// FIXME: implement where the value is not given as a tensor (int, ints, float, floats, etc.)
    else
    {
        return OperatorExecuteResult::ATTRIBUTE_ERROR;
    }
}