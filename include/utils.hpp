#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>

#ifdef USE_HIP
#include <hip/hip_runtime.h>

#define hipErrorCheck(ans)                  \
   {                                        \
      hipAssert((ans), __FILE__, __LINE__); \
   }
inline void hipAssert(hipError_t code, const char *file, int line, bool abort = true)
{
   if (code != hipSuccess)
   {
      fprintf(stderr, "HIP Error: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort)
         exit(code);
   }
}

// launch kernel and check for errors
// using hipErrorCheck(hipGetLastError());
#define hipKernelLaunchCheck(KernelCall) \
   {                                     \
      KernelCall;                        \
      hipErrorCheck(hipGetLastError());  \
   }

#endif // USE_HIP

#endif // UTILS_HPP