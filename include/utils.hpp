#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>

#ifdef USE_HIP
#include <hip/hip_runtime.h>

#define hipErrorCheck(ans) \
   hipAssert((ans), __FILE__, __LINE__);

inline void hipAssert(hipError_t code, const char *file, int line, bool abort = true)
{
   if (code != hipSuccess)
   {
      fprintf(stderr, "HIP Error: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort)
         exit(code);
   }
}

#define hipKernelLaunchCheck(KernelCall)     \
   {                                         \
      KernelCall;                            \
      hipErrorCheck(hipDeviceSynchronize()); \
      hipErrorCheck(hipGetLastError());      \
   }

#define CeilDiv(a, b) ((a + b - 1) / b)
#define MAX_DIMS 8

#endif // USE_HIP

#endif // UTILS_HPP