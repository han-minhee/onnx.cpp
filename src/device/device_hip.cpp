#ifdef USE_HIP

#include "device/hip.hpp"
#include <hip/hip_runtime.h>

HipDevice::HipDevice(int device_id) : device_id(device_id)
{
    hipSetDevice(device_id);
    hipStreamCreate(&stream);
}

HipDevice::~HipDevice()
{
    hipStreamDestroy(stream);
}

void HipDevice::setDeviceId(int id)
{
    device_id = id;
    hipSetDevice(device_id);
}

#endif // USE_HIP