#include "device/device.hpp"
#include <string>

DeviceType Device::getType() const
{
    return DeviceType::CPU; // Default implementation or throw an exception
}

int Device::getDeviceIndex()
{
    return 0; // Default implementation or throw an exception
}

void Device::setDeviceIndex(int index)
{
    // Default implementation or throw an exception
}

std::string DeviceUtils::DeviceTypeToString(DeviceType type)
{
    switch (type)
    {
    case DeviceType::CPU:
        return "CPU";
#ifdef USE_HIP
    case DeviceType::HIP:
        return "HIP";
#endif
    default:
        return "UNKNOWN";
    }
}