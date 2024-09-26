#ifdef USE_HIP

#include "operator/operators.hpp"
#include <iostream>
#include <cstring>

namespace HIP_OP
{
    OperatorExecuteResult ConstantOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                        const std::unordered_map<std::string, Node::AttributeValue> &attributes)
    {
        if (attributes.count("value"))
        {
            const Tensor &source_tensor = std::get<Tensor>(attributes.at("value"));

            // Allocate buffer for the output tensor if necessary
            if (!outputs[0]->getBuffer() || outputs[0]->getNumElements() != source_tensor.getNumElements())
            {
                outputs[0]->allocateBuffer(source_tensor.getDataType(), source_tensor.getNumElements());
            }

            // Set output tensor's data type and dimensions
            outputs[0]->setDataType(source_tensor.getDataType());
            outputs[0]->reshape(source_tensor.getDims());

            // Copy data from the source tensor to the output tensor
            const void *source_data_ptr = source_tensor.getBuffer()->getDataPointer();
            void *dest_data_ptr = outputs[0]->getBuffer()->getDataPointer();
            size_t data_size = source_tensor.getBuffer()->getSizeInBytes();

            std::memcpy(dest_data_ptr, source_data_ptr, data_size);

            return OperatorExecuteResult::SUCCESS;
        }
        // Handle other types (int, ints, float, floats, etc.)
        else if (attributes.count("int_value"))
        {
            int int_value = std::get<int64_t>(attributes.at("int_value"));
            outputs[0]->allocateBuffer(TensorDataType::INT64, 1); // Assuming a single int value
            int *data_ptr = outputs[0]->data<int>();
            *data_ptr = int_value;
            return OperatorExecuteResult::SUCCESS;
        }
        else if (attributes.count("float_value"))
        {
            float float_value = std::get<float>(attributes.at("float_value"));
            outputs[0]->allocateBuffer(TensorDataType::FLOAT32, 1); // Assuming a single float value
            float *data_ptr = outputs[0]->data<float>();
            *data_ptr = float_value;
            return OperatorExecuteResult::SUCCESS;
        }
        else if (attributes.count("ints_value"))
        {
            const std::vector<int64_t> &ints_value = std::get<std::vector<int64_t>>(attributes.at("ints_value"));
            outputs[0]->allocateBuffer(TensorDataType::INT64, ints_value.size());
            int *data_ptr = outputs[0]->data<int>();
            std::copy(ints_value.begin(), ints_value.end(), data_ptr);
            outputs[0]->reshape({ints_value.size()});
            return OperatorExecuteResult::SUCCESS;
        }
        else if (attributes.count("floats_value"))
        {
            const std::vector<float> &floats_value = std::get<std::vector<float>>(attributes.at("floats_value"));
            outputs[0]->allocateBuffer(TensorDataType::FLOAT32, floats_value.size());
            float *data_ptr = outputs[0]->data<float>();
            std::copy(floats_value.begin(), floats_value.end(), data_ptr);
            outputs[0]->reshape({floats_value.size()});
            return OperatorExecuteResult::SUCCESS;
        }
        else
        {
            return OperatorExecuteResult::ATTRIBUTE_ERROR;
        }
    }
}

#endif