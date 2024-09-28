#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

namespace HIP_OP
{

    OperatorExecuteResult ConstantOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                        const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {

        if (attributes.count("value"))
        {
            const Tensor &source_tensor = std::get<Tensor>(attributes.at("value"));
            outputs[0]->copyFrom(source_tensor);


            return OperatorExecuteResult::SUCCESS;
        }
        // Handle other types (int, ints, float, floats, etc.)
        else if (attributes.count("int_value"))
        {

            return OperatorExecuteResult::SUCCESS;
        }
        else if (attributes.count("float_value"))
        {

            return OperatorExecuteResult::SUCCESS;
        }
        else if (attributes.count("ints_value"))
        {

            return OperatorExecuteResult::SUCCESS;
        }
        else if (attributes.count("floats_value"))
        {

            return OperatorExecuteResult::SUCCESS;
        }
        else
        {
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        }

        return OperatorExecuteResult::SUCCESS;
    }
};

#endif
