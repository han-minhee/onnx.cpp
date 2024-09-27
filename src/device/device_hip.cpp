#include "utils.hpp"
#ifdef USE_HIP

#include "device/device_hip.hpp"
#include <hip/hip_runtime.h>

HipDevice::HipDevice(int device_id) : device_id(device_id)
{
    hipErrorCheck(hipSetDevice(device_id));
    hipErrorCheck(hipStreamCreate(&stream));
}

HipDevice::~HipDevice()
{
    hipErrorCheck(hipStreamDestroy(stream));
}

hipStream_t HipDevice::getStream() const
{
    return stream;
}

void HipDevice::setStream(hipStream_t s)
{
    stream = s;
}

int HipDevice::getDeviceIndex() const 
{
    return device_id;
}

void HipDevice::setDeviceIndex(int id) 
{
    device_id = id;
    hipErrorCheck(hipSetDevice(device_id));
}

DeviceType HipDevice::getType() const
{
    return DeviceType::HIP;  // Assuming 'DeviceType::HIP' is defined in your enum
}

std::string HipDevice::toString() const
{
    return "HipDevice(ID: " + std::to_string(device_id) + ")";
}

#endif // USE_HIP
