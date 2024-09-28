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
            // make a single element tensor with int64_t
            int64_t value = std::get<int64_t>(attributes.at("int_value"));
            outputs[0]->setData<int64_t>({value});
            return OperatorExecuteResult::SUCCESS;
        }
        else if (attributes.count("float_value"))
        {
            float value = std::get<float>(attributes.at("float_value"));
            outputs[0]->setData<float>({value});
            return OperatorExecuteResult::SUCCESS;
        }
        else if (attributes.count("ints_value"))
        {
            // int64_t vector
            std::vector<int64_t> values = std::get<std::vector<int64_t>>(attributes.at("ints_value"));
            outputs[0]->setData<int64_t>(values);

            return OperatorExecuteResult::SUCCESS;
        }
        else if (attributes.count("floats_value"))
        {
            // float vector
            std::vector<float> values = std::get<std::vector<float>>(attributes.at("floats_value"));
            outputs[0]->setData<float>(values);
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
