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

    // Define the += operator
#ifdef USE_HIP
    __host__ __device__
#endif
        half_t &
        operator+=(const half_t &rhs)
    {
#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
        value = __hadd(value, rhs.value); // Use HIP intrinsic for addition
#else
        value = hf::half(static_cast<float>(*this) + static_cast<float>(rhs));
#endif
#else
        value = hf::half(static_cast<float>(*this) + static_cast<float>(rhs));
#endif
        return *this;
    }

    // Define the -= operator
#ifdef USE_HIP
    __host__ __device__
#endif
        half_t &
        operator-=(const half_t &rhs)
    {
#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
        value = __hsub(value, rhs.value); // Use HIP intrinsic for subtraction
#else
        value = hf::half(static_cast<float>(*this) - static_cast<float>(rhs));
#endif
#else
        value = hf::half(static_cast<float>(*this) - static_cast<float>(rhs));
#endif
        return *this;
    }

    // Define the *= operator
#ifdef USE_HIP
    __host__ __device__
#endif
        half_t &
        operator*=(const half_t &rhs)
    {
#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
        value = __hmul(value, rhs.value); // Use HIP intrinsic for multiplication
#else
        value = hf::half(static_cast<float>(*this) * static_cast<float>(rhs));
#endif
#else
        value = hf::half(static_cast<float>(*this) * static_cast<float>(rhs));
#endif
        return *this;
    }

    // Define the /= operator
#ifdef USE_HIP
    __host__ __device__
#endif
        half_t &
        operator/=(const half_t &rhs)
    {
#ifdef USE_HIP
#ifdef __HIP_DEVICE_COMPILE__
        value = __hdiv(value, rhs.value); // Use HIP intrinsic for division
#else
        value = hf::half(static_cast<float>(*this) / static_cast<float>(rhs));
#endif
#else
        value = hf::half(static_cast<float>(*this) / static_cast<float>(rhs));
#endif
        return *this;
    }

    // Define the assignment operator to prevent compiler issues
#ifdef USE_HIP
    __host__ __device__
#endif
        half_t &
        operator=(const half_t &rhs)
    {
        value = rhs.value;
        return *this;
    }

    // Explicit conversions (already provided)
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