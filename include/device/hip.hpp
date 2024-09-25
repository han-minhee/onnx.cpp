#ifdef USE_HIP
#ifndef HIP_HPP
#define HIP_HPP
#include <hip/hip_runtime.h>

class HipDevice : public Device
{
};

#endif // HIP_HPP
#endif // USE_SYCL