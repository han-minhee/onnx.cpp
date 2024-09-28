#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

namespace HIP_OP
{

    template <typename T>
    OperatorExecuteResult ExecuteTranspose()
    {
    }

    OperatorExecuteResult TransposeOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                         const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        const Tensor &input = inputs[0];
        Tensor *output = outputs[0];

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif
