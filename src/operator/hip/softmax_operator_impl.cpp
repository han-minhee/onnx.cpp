#ifdef USE_HIP
#include "operator/operators.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

namespace HIP_OP
{

    OperatorExecuteResult SoftmaxOperatorImpl::execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs,
                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {
        // check if the device is a HIP device
        if (device->getType() != DeviceType::HIP)
        {
            throw std::runtime_error("Device is not a HIP device");
        }
        else
        {
            std::cout << "Device is a HIP device" << std::endl;
        }

        return OperatorExecuteResult::NOT_IMPLEMENTED;
    }
};

#endif
