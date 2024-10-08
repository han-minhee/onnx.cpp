#include "operator/operators.hpp"
#include <iostream>

namespace CPU_OP
{
        OperatorExecuteResult ShapeOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        const Tensor &input = inputs[0];
        Tensor *output = outputs[0];
        std::vector<size_t> dims = input.getDims();
        std::vector<int64_t> dims_int64(dims.begin(), dims.end());
        output->setData<int64_t>(dims_int64);
        return OperatorExecuteResult::SUCCESS;
    }
}
