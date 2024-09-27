#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <string>
#include "enums.hpp"

namespace DeviceUtils
{
    std::string DeviceTypeToString(DeviceType type);
}

class Device
{
public:
    Device() {}
    virtual ~Device() {}
    virtual DeviceType getType() const = 0;
    virtual std::string toString() const = 0;
    virtual int getDeviceIndex() const = 0;
    virtual void setDeviceIndex(int index) = 0;

private:
};

#endif // DEVICE_HPP
