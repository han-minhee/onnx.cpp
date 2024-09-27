#ifdef USE_HIP
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "tensor/tensor.hpp"
#include "operator/operators.hpp"
#include "operator/operator_registry.hpp"

#include "device/device.hpp"
#include "device/device_hip.hpp"

#include "utils.hpp"

__global__ void helloWorldKernel()
{
    printf("Hello, World from GPU!\n");
}

TEST(HIP_TEST, HelloWorld)
{
    HipDevice hipDevice = HipDevice(0);
    hipKernelLaunchCheck(hipLaunchKernelGGL(helloWorldKernel, dim3(1), dim3(1), 0, hipDevice.getStream()));

    hipDevice.synchronize();
}

#endif // USE_HIP