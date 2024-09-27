#ifndef DEVICE_HIP_HPP
#define DEVICE_HIP_HPP

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include "device/device.hpp"

class HipDevice : public Device
{
public:
    HipDevice(int device_id = 0);
    ~HipDevice();

    int getDeviceIndex() const override;
    void setDeviceIndex(int id) override;
    hipStream_t getStream() const;

    void setStream(hipStream_t s);
    void synchronize() const;

    DeviceType getType() const override;         // Implement getType()
    std::string toString() const override;       // Implement toString()

private:
    int device_id;
    hipStream_t stream;
};

#endif // USE_HIP

#endif // DEVICE_HIP_HPP
