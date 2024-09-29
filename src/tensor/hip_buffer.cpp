#ifdef USE_HIP

#include <stdexcept>

#include <sstream>
#include "device/device_hip.hpp"
#include "tensor/buffer.hpp"
#include "tensor/tensor_utils.hpp"

#include "utils.hpp"

HipBuffer::HipBuffer(TensorDataType data_type, size_t num_elements, HipDevice *device)
    : data_type_(data_type), num_elements_(num_elements), device_(device)
{
    size_t element_size = TensorUtils::getDataTypeSize(data_type_);
    size_in_bytes_ = element_size * num_elements_;
    hipError_t result = hipMalloc(&data_, size_in_bytes_);
    if (result != hipSuccess)
    {
        throw std::runtime_error("Failed to allocate memory for HipBuffer");
    }
}

HipBuffer::~HipBuffer()
{
    // hipErrorCheck(hipFree(data_));
}

Device *HipBuffer::getDevice() const
{
    return device_;
}

DeviceType HipBuffer::getDeviceType() const
{
    return DeviceType::HIP;
}

int HipBuffer::getDeviceId() const
{
    return device_->getDeviceIndex();
}

void *HipBuffer::getDataPointer()
{
    return data_;
}

const void *HipBuffer::getDataPointer() const
{
    return data_;
}

TensorDataType HipBuffer::getDataType() const
{
    return data_type_;
}

size_t HipBuffer::getNumElements() const
{
    return num_elements_;
}

size_t HipBuffer::getSizeInBytes() const
{
    return size_in_bytes_;
}

void HipBuffer::resize(size_t num_elements)
{
    size_t element_size = TensorUtils::getDataTypeSize(data_type_);
    size_t new_size_in_bytes = element_size * num_elements;
    hipError_t result = hipMalloc(&data_, new_size_in_bytes);
    if (result != hipSuccess)
    {
        throw std::runtime_error("Failed to reallocate memory for HipBuffer");
    }
    num_elements_ = num_elements;
    size_in_bytes_ = new_size_in_bytes;
}

Buffer *HipBuffer::to(Device *device)
{
    switch (device->getType())
    {
    case DeviceType::CPU:
    {
        // Create a new CpuBuffer
        CpuBuffer *cpuBuffer = new CpuBuffer(data_type_, num_elements_, dynamic_cast<CpuDevice *>(device));
        hipErrorCheck(hipMemcpy(cpuBuffer->getDataPointer(), data_, size_in_bytes_, hipMemcpyDeviceToHost));
        // Free the old HIP data
        hipErrorCheck(hipFree(data_));
        data_ = nullptr;

        // Return the new CpuBuffer instance
        return cpuBuffer;
    }

    default:
        throw std::runtime_error("Unsupported device type");
    }
}

BufferOperationResult HipBuffer::offload(Device *device)
{
    switch (device->getType())
    {
    case DeviceType::CPU:
    {
        return BufferOperationResult::NOT_IMPLEMENTED;
    }

    default:
        return BufferOperationResult::NOT_IMPLEMENTED;
    }
}

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
    case TensorDataType::UINT8:
        printData(static_cast<uint8_t *>(host_data));
        break;

    // custom types
    case TensorDataType::FLOAT16:
        printData(static_cast<half_t *>(host_data));
        break;
        
    default:
        oss << "Unsupported data type";
    }

    oss << "]";
    free(host_data);
    return oss.str();
}

void HipBuffer::copyFrom(const Buffer *src)
{
    switch (src->getDeviceType())
    {
    case DeviceType::CPU:
    {
        const void *src_data = src->getDataPointer();
        hipErrorCheck(hipMemcpy(data_, src_data, size_in_bytes_, hipMemcpyHostToDevice));
        break;
    }
    case DeviceType::HIP:
    {
        const HipBuffer *hip_src = dynamic_cast<const HipBuffer *>(src);
        hipErrorCheck(hipMemcpy(data_, hip_src->getDataPointer(), size_in_bytes_, hipMemcpyDeviceToDevice));
        break;
    }

    default:
        throw std::runtime_error("Unsupported device type");
    }
}

#endif // USE_HIP
