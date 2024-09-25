#ifdef USE_SYCL
#ifndef SYCL_HPP
#define SYCL_HPP

// currently, it's just a place holder to show how multiple backends can be supported
#include <sycl/sycl.hpp>

class SyclDevice : public Device
{
    
};

#endif // SYCL_HPP
#endif // USE_SYCL