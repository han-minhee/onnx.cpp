#include "device/device.hpp"
#include <string>

class CpuDevice : public Device
{
public:
    DeviceType getType() const override
    {
        return DeviceType::CPU;
    }

    std::string toString() const override
    {
        // std::ifstream cpuinfo("/proc/cpuinfo");
        return "CPU";
    }
};