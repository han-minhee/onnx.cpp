#ifndef HALF_T_HPP
#define HALF_T_HPP

#include "half.hpp" // Include the half-precision floating-point library
#include <iostream>
#include <cmath>
#include <cstdint> // For int64_t
#include <cstddef> // For size_t

// Include HIP headers only if USE_HIP is defined
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

// Namespace alias for the half library
namespace hf = half_float;

// Custom half wrapper class
struct half_t
{
    // Use HIP __host__ and __device__ annotations only if USE_HIP is defined
#ifdef USE_HIP
    __host__ __device__
#endif
    half_t() : value(0)
    {
    }

    // Constructor from float
#ifdef USE_HIP
    __host__ __device__
#endif
    half_t(float f)
    {
#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
        value = __float2half(f);
#else
        value = hf::half(f);
#endif
#else
        value = hf::half(f);
#endif
    }

    // Conversion to float
#ifdef USE_HIP
    __host__ __device__
#endif
    operator float() const
    {
#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
        return __half2float(value);
#else
        return static_cast<float>(value);
#endif
#else
        return static_cast<float>(value);
#endif
    }

    // Conversion to int
#ifdef USE_HIP
    __host__ __device__
#endif
        explicit
        operator int() const
    {
#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
        return static_cast<int>(__half2float(value));
#else
        return static_cast<int>(static_cast<float>(value));
#endif
#else
        return static_cast<int>(static_cast<float>(value));
#endif
    }

    // Conversion to double
#ifdef USE_HIP
    __host__ __device__
#endif
        explicit
        operator double() const
    {
#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
        return static_cast<double>(__half2float(value));
#else
        return static_cast<double>(static_cast<float>(value));
#endif
#else
        return static_cast<double>(static_cast<float>(value));
#endif
    }

    // Conversion to int64_t
#ifdef USE_HIP
    __host__ __device__
#endif
        explicit
        operator int64_t() const
    {
#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
        return static_cast<int64_t>(__half2float(value));
#else
        return static_cast<int64_t>(static_cast<float>(value));
#endif
#else
        return static_cast<int64_t>(static_cast<float>(value));
#endif
    }

    // Conversion to size_t
#ifdef USE_HIP
    __host__ __device__
#endif
        explicit
        operator size_t() const
    {
#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
        return static_cast<size_t>(__half2float(value));
#else
        return static_cast<size_t>(static_cast<float>(value));
#endif
#else
        return static_cast<size_t>(static_cast<float>(value));
#endif
    }

#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
    __half value; // Device-side representation
#else
    hf::half value; // Host-side representation
#endif
#else
    hf::half value; // Use half_float's half when HIP is not in use
#endif
};

#endif // HALF_T_HPP