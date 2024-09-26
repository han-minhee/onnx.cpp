#include "tensor/buffer.hpp"
#include "tensor/tensor_utils.hpp"
#include <stdexcept>

CpuBuffer::CpuBuffer(TensorDataType data_type, size_t num_elements)
    : data_type_(data_type), num_elements_(num_elements)
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

BufferOperationResult CpuBuffer::toHost()
{

    return BufferOperationResult::SUCCESS;
}

BufferOperationResult CpuBuffer::to(DeviceType deviceType)
{
    if (deviceType == DeviceType::CPU)
    {
        return BufferOperationResult::SUCCESS;
    }
    else
    {

        return BufferOperationResult::NOT_IMPLEMENTED;
    }
}

BufferOperationResult CpuBuffer::offload(DeviceType deviceType)
{

    return BufferOperationResult::NOT_IMPLEMENTED;
}
