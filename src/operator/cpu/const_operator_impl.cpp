#include "operator/operators.hpp"
#include <iostream>

namespace CPU_OP
{
    OperatorExecuteResult ConstantOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
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
}