#include "device/device.hpp"
#include <string>

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