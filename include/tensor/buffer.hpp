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

    // Pure virtual methods for data access
    virtual void *data() = 0;
    virtual const void *data() const = 0;
    virtual size_t size() const = 0;

    // Virtual method to resize the buffer
    virtual void resize(size_t newSize) = 0;

    virtual BufferOperationResult toHost()
    {
        return to(DeviceType::CPU);
    };

    // this will permenantly move the buffer to the device
    virtual BufferOperationResult to(DeviceType deviceType);

    // this will temporarily move the buffer to the device
    // and when needed, it will move it back to the original device
    virtual BufferOperationResult offload(DeviceType deviceType);

};

class CpuBuffer : public Buffer
{
public:
    CpuBuffer(size_t size = 0);
    ~CpuBuffer() override;

    void *data() override;
    const void *data() const override;
    size_t size() const override;

    void resize(size_t newSize) override;

private:
    std::vector<uint8_t> storage;
};

#ifdef USE_HIP
class HipBuffer : public Buffer
{
public:
    HipBuffer(size_t size = 0);
    ~HipBuffer() override;

    void *data() override;
    const void *data() const override;
    size_t size() const override;

    void resize(size_t newSize) override;

private:
    void *devicePtr;
    size_t bufferSize;
};
#endif // USE_HIP

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
class SyclBuffer : public Buffer
{
public:
    SyclBuffer(size_t size = 0);
    ~SyclBuffer() override;

    void *data() override;
    const void *data() const override;
    size_t size() const override;

    void resize(size_t newSize) override;

private:
    cl::sycl::buffer<uint8_t, 1> buffer;
    size_t bufferSize;
};
#endif // USE_SYCL

#endif // BUFFER_HPP
