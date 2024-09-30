#include "tensor/buffer.hpp"

void Buffer::copyFrom(const Buffer *src)
{
    if (this->getSizeInBytes() != src->getSizeInBytes())
    {
        throw std::runtime_error("Buffer size mismatch.");
    }

    switch (this->getDeviceType())
    {
    case DeviceType::CPU:
        dynamic_cast<CpuBuffer *>(this)->copyFrom(dynamic_cast<const CpuBuffer *>(src));
        break;
#ifdef USE_HIP
    case DeviceType::HIP:
        dynamic_cast<HipBuffer *>(this)->copyFrom(dynamic_cast<const HipBuffer *>(src));
        break;
#endif
    default:
        throw std::runtime_error("Unsupported device type");
    }
}

std::shared_ptr<Buffer> Buffer::create(Device *device, TensorDataType data_type, size_t num_elements)
{
    if (device == nullptr)
    {
        throw std::invalid_argument("Device pointer is null");
    }

    switch (device->getType())
    {
    case DeviceType::CPU:
    {
        // what if the device is not a CPU device?
        // handle where the device is not a CPU device
        return std::make_shared<CpuBuffer>(data_type, num_elements, dynamic_cast<CpuDevice *>(device));
    }

#ifdef USE_HIP
    case DeviceType::HIP:
        return std::make_shared<HipBuffer>(data_type, num_elements, dynamic_cast<HipDevice *>(device));
#endif

    default:
        throw std::runtime_error("Unsupported device type");
    }
}

template <typename T>
void Buffer::setData(const std::vector<T> &data)
{
    switch (this->getDeviceType())
    {
    case DeviceType::CPU:
        dynamic_cast<CpuBuffer *>(this)->setData(data);
        break;
#ifdef USE_HIP
    case DeviceType::HIP:
        dynamic_cast<HipBuffer *>(this)->setData(data);
        break;
#endif
    default:
        throw std::runtime_error("Unsupported device type");
    }
}

template <typename T>
void CpuBuffer::setData(const std::vector<T> &data)
{
    if (sizeof(T) * data.size() != this->getSizeInBytes())
    {
        // print the size of the data and the size of the buffer
        std::cerr << "Buffer size mismatch." << std::endl;
        std::cerr << "Data size: " << sizeof(T) * data.size() << std::endl;
        std::cerr << "Buffer size: " << this->getSizeInBytes() << std::endl;
    }
    std::memcpy(getDataPointer(), data.data(), data.size() * sizeof(T));
}

#ifdef USE_HIP

template <typename T>
void HipBuffer::setData(const std::vector<T> &data)
{
    if (sizeof(T) * data.size() != this->getSizeInBytes())
    {
        throw std::runtime_error("Data size mismatch.");
    }
    hipError_t result = hipMemcpy(getDataPointer(), data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice);
    if (result != hipSuccess)
    {
        throw std::runtime_error("HIP memory copy failed.");
    }
}
#endif

// Explicit instantiation of the setData template for supported types
template void Buffer::setData<float>(const std::vector<float> &);
template void Buffer::setData<double>(const std::vector<double> &);
template void Buffer::setData<int>(const std::vector<int> &);
template void Buffer::setData<long>(const std::vector<long> &);
template void Buffer::setData<signed char>(const std::vector<signed char> &);
template void Buffer::setData<unsigned char>(const std::vector<unsigned char> &);

// custom types
template void Buffer::setData<half_t>(const std::vector<half_t> &);