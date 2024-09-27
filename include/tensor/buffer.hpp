// Buffer.hpp
#ifndef BUFFER_HPP
#define BUFFER_HPP

#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring> // For memcpy
#include "enums.hpp"
#include "device/device.hpp"
#include "device/device_cpu.hpp"

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include "device/hip.hpp"
#endif

class Buffer
{
public:
    virtual ~Buffer() = default;

    virtual Device *getDevice() const = 0;
    virtual DeviceType getDeviceType() const = 0;
    virtual int getDeviceId() const = 0;

    virtual void *getDataPointer() = 0;
    virtual const void *getDataPointer() const = 0;

    virtual TensorDataType getDataType() const = 0;
    virtual size_t getNumElements() const = 0;
    virtual size_t getSizeInBytes() const = 0;

    virtual void resize(size_t num_elements) = 0;

    virtual Buffer *to(Device *device) = 0;
    virtual BufferOperationResult offload(Device *device) = 0;

    template <typename T>
    void setData(const std::vector<T> &data);

    static std::shared_ptr<Buffer> create(Device *device, TensorDataType data_type, size_t num_elements);

    virtual std::string toString(size_t max_elements = 5) const = 0;
};

class CpuBuffer : public Buffer
{
public:
    CpuBuffer(TensorDataType data_type, size_t num_elements, CpuDevice *device) ;
    ~CpuBuffer() override;

    Device *getDevice() const override;
    DeviceType getDeviceType() const override;
    int getDeviceId() const override;

    void *getDataPointer() override;
    const void *getDataPointer() const override;

    TensorDataType getDataType() const override;
    size_t getNumElements() const override;
    size_t getSizeInBytes() const override;

    void resize(size_t num_elements) override;

    Buffer *to(Device *device) override;
    BufferOperationResult offload(Device *device) override;

    template <typename T>
    void setData(const std::vector<T> &data);

    std::string toString(size_t max_elements = 5) const override;

private:
    CpuDevice *device_;
    void *data_;
    TensorDataType data_type_;
    size_t num_elements_;
    size_t size_in_bytes_;
};

#ifdef USE_HIP

class HipBuffer : public Buffer
{
public:
    HipBuffer(TensorDataType data_type, size_t num_elements, HipDevice *device);
    ~HipBuffer() override;

    Device *getDevice() const override;
    DeviceType getDeviceType() const override;
    int getDeviceId() const override;

    void *getDataPointer() override;
    const void *getDataPointer() const override;

    TensorDataType getDataType() const override;
    size_t getNumElements() const override;
    size_t getSizeInBytes() const override;

    void resize(size_t num_elements) override;

    Buffer *to(Device *device) override;
    BufferOperationResult offload(Device *device) override;

    template <typename T>
    void setData(const std::vector<T> &data);

    std::string toString(size_t max_elements = 5) const override;

private:
    HipDevice *device_;
    void *data_;
    TensorDataType data_type_;
    size_t num_elements_;
    size_t size_in_bytes_;
};

#endif // USE_HIP
#endif // BUFFER_HPP
