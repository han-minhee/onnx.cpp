#ifndef BUFFER_HPP
#define BUFFER_HPP

#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include "enums.hpp"

class Buffer
{
public:
    virtual ~Buffer() = default;

    virtual DeviceType getDeviceType() const = 0;
    virtual int getDeviceId() const = 0;

    virtual void *getDataPointer() = 0;
    virtual const void *getDataPointer() const = 0;

    virtual TensorDataType getDataType() const = 0;
    virtual size_t getNumElements() const = 0;
    virtual size_t getSizeInBytes() const = 0;

    virtual void resize(size_t num_elements) = 0;

    virtual BufferOperationResult toHost() = 0;
    virtual BufferOperationResult to(DeviceType deviceType) = 0;
    virtual BufferOperationResult offload(DeviceType deviceType) = 0;
};

class CpuBuffer : public Buffer
{
public:
    CpuBuffer(TensorDataType data_type, size_t num_elements);
    ~CpuBuffer() override;

    DeviceType getDeviceType() const override;
    int getDeviceId() const override;

    void *getDataPointer() override;
    const void *getDataPointer() const override;

    TensorDataType getDataType() const override;
    size_t getNumElements() const override;
    size_t getSizeInBytes() const override;

    void resize(size_t num_elements) override;

    BufferOperationResult toHost() override;
    BufferOperationResult to(DeviceType deviceType) override;
    BufferOperationResult offload(DeviceType deviceType) override;

private:
    void *data_;
    TensorDataType data_type_;
    size_t num_elements_;
    size_t size_in_bytes_;
};

#ifdef USE_HIP

class HipBuffer : public Buffer
{
public:
    HipBuffer(TensorDataType data_type, size_t num_elements);
    ~HipBuffer() override;

    DeviceType getDeviceType() const override;
    int getDeviceId() const override;

    void *getDataPointer() override;
    const void *getDataPointer() const override;

    TensorDataType getDataType() const override;
    size_t getNumElements() const override;
    size_t getSizeInBytes() const override;

    void resize(size_t num_elements) override;

    BufferOperationResult toHost() override;
    BufferOperationResult to(DeviceType deviceType) override;
    BufferOperationResult offload(DeviceType deviceType) override;

private:
    void *data_;
    TensorDataType data_type_;
    size_t num_elements_;
    size_t size_in_bytes_;
};

#endif // USE_HIP
#endif // BUFFER_HPP

// #ifdef USE_SYCL
// #include <sycl/sycl.hpp>
// class SyclBuffer : public Buffer
// {
// public:
//     SyclBuffer(size_t size = 0);
//     ~SyclBuffer() override;

//     void *data() override;
//     const void *data() const override;
//     size_t size() const override;

//     void resize(size_t newSize) override;

// private:
//     cl::sycl::buffer<uint8_t, 1> buffer;
//     size_t bufferSize;
// };
// #endif // USE_SYCL