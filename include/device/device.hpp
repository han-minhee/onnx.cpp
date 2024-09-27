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
    virtual DeviceType getType() const;
    virtual std::string toString() const;
    virtual int getDeviceIndex();
    virtual void setDeviceIndex(int index);

private:
};

#endif // DEVICE_HPP