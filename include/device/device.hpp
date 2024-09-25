#ifndef DEVICE_HPP
#define DEVICE_HPP

enum class DeviceType
{
    CPU,
#ifdef USE_HIP
    HIP,
#endif

#ifdef USE_SYCL
    SYCL
#endif

};

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

private:
};

#endif // DEVICE_HPP