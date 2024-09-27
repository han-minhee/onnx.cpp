#ifdef USE_HIP

#include <stdexcept>

#include <sstream>
#include "device/hip.hpp"
#include "tensor/buffer.hpp"
#include "tensor/tensor_utils.hpp"



// Implementation of the toString method in HipBuffer
std::string HipBuffer::toString(size_t max_elements) const
{
    std::ostringstream oss;
    oss << "HipBuffer: dtype=" << TensorUtils::getDataTypeName(data_type_) << ", data=[";

    // Allocate temporary host memory
    void *host_data = malloc(size_in_bytes_);
    if (!host_data)
    {
        oss << "Error: Could not allocate memory.";
        return oss.str();
    }

    // Copy data from device to host
    hipError_t result = hipMemcpy(host_data, data_, size_in_bytes_, hipMemcpyDeviceToHost);
    if (result != hipSuccess)
    {
        oss << "Error: Could not copy data from device.";
        free(host_data);
        return oss.str();
    }

    // Helper lambda to handle data printing
    auto printData = [&](auto *data_ptr)
    {
        for (size_t i = 0; i < num_elements_ && i < max_elements; ++i)
        {
            oss << data_ptr[i];
            if (i < num_elements_ - 1 && i < max_elements - 1)
            {
                oss << ", ";
            }
        }
        if (num_elements_ > max_elements)
        {
            oss << "...";
        }
    };

    switch (data_type_)
    {
    case TensorDataType::FLOAT32:
        printData(static_cast<float *>(host_data));
        break;
    case TensorDataType::FLOAT64:
        printData(static_cast<double *>(host_data));
        break;
    case TensorDataType::INT32:
        printData(static_cast<int32_t *>(host_data));
        break;
    case TensorDataType::INT64:
        printData(static_cast<int64_t *>(host_data));
        break;
    case TensorDataType::INT8:
        printData(static_cast<int8_t *>(host_data));
        break;
    default:
        oss << "Unsupported data type";
    }

    oss << "]";
    free(host_data);
    return oss.str();
}

#endif // USE_HIP
