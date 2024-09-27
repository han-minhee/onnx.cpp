#ifndef DEVICE_CPU_HPP
#define DEVICE_CPU_HPP

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

    int getDeviceIndex() const override
    {
        return 0;
    }

    void setDeviceIndex(int index) override
    {
        // Do nothing
    }
};

#endif // DEVICE_CPU_HPP