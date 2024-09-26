#ifdef USE_HIP
#ifndef HIP_HPP
#define HIP_HPP
#include <hip/hip_runtime.h>
#include "device/device.hpp"

class HipDevice : public Device
{
public:
    HipDevice(int device_id = 0);
    ~HipDevice();

    int getDeviceId() const { return device_id; }
    hipStream_t getStream() const { return stream; }
    void setDeviceId(int id);
    void setStream(hipStream_t s);
    void synchronize() const;

private:
    int device_id;
    hipStream_t stream;
    
};

#endif // HIP_HPP
#endif // USE_SYCL