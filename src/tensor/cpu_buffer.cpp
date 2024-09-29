#include "tensor/buffer.hpp"
#include "tensor/tensor_utils.hpp"
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cstring>
#include "utils.hpp"

#ifdef USE_HIP
#include "device/device_hip.hpp"
#endif // USE_HIP

CpuBuffer::CpuBuffer(TensorDataType data_type, size_t num_elements, CpuDevice *device)
    : data_type_(data_type), num_elements_(num_elements), device_(device)
{
    size_t element_size = TensorUtils::getDataTypeSize(data_type_);
    size_in_bytes_ = element_size * num_elements_;
    data_ = malloc(size_in_bytes_);
    if (!data_)
    {
        throw std::runtime_error("Failed to allocate memory for CpuBuffer");
    }
}

CpuBuffer::~CpuBuffer()
{
    free(data_);
}

Device *CpuBuffer::getDevice() const
{
    return device_;
}

DeviceType CpuBuffer::getDeviceType() const
{
    return DeviceType::CPU;
}

int CpuBuffer::getDeviceId() const
{
    return 0;
}

void *CpuBuffer::getDataPointer()
{
    return data_;
}

const void *CpuBuffer::getDataPointer() const
{
    return data_;
}

TensorDataType CpuBuffer::getDataType() const
{
    return data_type_;
}

size_t CpuBuffer::getNumElements() const
{
    return num_elements_;
}

size_t CpuBuffer::getSizeInBytes() const
{
    return size_in_bytes_;
}

void CpuBuffer::resize(size_t num_elements)
{
    size_t element_size = TensorUtils::getDataTypeSize(data_type_);
    size_t new_size_in_bytes = element_size * num_elements;
    void *new_data = realloc(data_, new_size_in_bytes);
    if (!new_data)
    {
        throw std::runtime_error("Failed to reallocate memory for CpuBuffer");
    }
    data_ = new_data;
    num_elements_ = num_elements;
    size_in_bytes_ = new_size_in_bytes;
}

Buffer *CpuBuffer::to(Device *device)
{
    DeviceType deviceType = device->getType();

    switch (deviceType)
    {
    case DeviceType::CPU:
        return this; // Already a CPU buffer, return the current instance

#ifdef USE_HIP
    case DeviceType::HIP:
    {
        HipDevice *hipDevice = dynamic_cast<HipDevice *>(device);
        if (!hipDevice)
        {
            return nullptr; // Handle this error appropriately in your code
        }

        // Create a new HipBuffer
        HipBuffer *hipBuffer = new HipBuffer(data_type_, num_elements_, hipDevice);
        hipErrorCheck(hipMemcpy(hipBuffer->getDataPointer(), data_, size_in_bytes_, hipMemcpyHostToDevice));

        // Free the old CPU data
        free(data_);
        data_ = nullptr;

        // Return the new HipBuffer instance
        return hipBuffer;
    }
#endif

    default:
        return nullptr; // Unsupported device type
    }
}

BufferOperationResult CpuBuffer::offload(Device *device)
{
    return BufferOperationResult::NOT_IMPLEMENTED;
}

// Implementation of the toString method in CpuBuffer
std::string CpuBuffer::toString(size_t max_elements) const
{
    std::ostringstream oss;
    oss << "CpuBuffer: dtype=" << TensorUtils::getDataTypeName(data_type_) << ", data=[";

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
        printData(static_cast<float *>(data_));
        break;
    case TensorDataType::FLOAT64:
        printData(static_cast<double *>(data_));
        break;
    case TensorDataType::INT32:
        printData(static_cast<int32_t *>(data_));
        break;
    case TensorDataType::INT64:
        printData(static_cast<int64_t *>(data_));
        break;
    case TensorDataType::INT8:
        printData(static_cast<int8_t *>(data_));
        break;
    case TensorDataType::UINT8:
        printData(static_cast<uint8_t *>(data_));
        break;

    // custom types
    case TensorDataType::FLOAT16:
        printData(static_cast<half_t *>(data_));
        break;
    default:
        oss << "Unsupported data type";
    }

    oss << "]";
    return oss.str();
}

void CpuBuffer::copyFrom(const Buffer *src)
{
    switch (src->getDeviceType())
    {
    case DeviceType::CPU:
    {
        const void *src_data = src->getDataPointer();
        std::memcpy(data_, src_data, size_in_bytes_);
        break;
    }

#ifdef USE_HIP
    case DeviceType::HIP:
    {
        const HipBuffer *hip_src = dynamic_cast<const HipBuffer *>(src);
        hipErrorCheck(hipMemcpy(data_, hip_src->getDataPointer(), size_in_bytes_, hipMemcpyDeviceToHost));
        break;
    }
#endif

    default:
        throw std::runtime_error("Unsupported device type");
    }
}
