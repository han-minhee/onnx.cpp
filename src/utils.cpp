#include "utils.hpp"

#ifdef USE_HIP

// void hipErrorCheck(hipError_t result)
// {
//     if (result != hipSuccess)
//     {
//         std::cerr << "Error: " << hipGetErrorString(result) << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

// template <typename T>
// void debugPrint(const void *device_ptr, size_t num_elements, const std::string &name)
// {
//     std::vector<T> host_data(num_elements);
//     hipMemcpy(host_data.data(), device_ptr, num_elements * sizeof(T), hipMemcpyDeviceToHost);

//     std::cout << "First 5 elements of " << name << ": ";
//     for (size_t i = 0; i < std::min(num_elements, size_t(5)); ++i)
//     {
//         std::cout << host_data[i] << " ";
//     }
//     std::cout << std::endl;

//     // last 5 elements
//     std::cout << "Last 5 elements of " << name << ": ";
//     for (size_t i = std::max(num_elements, size_t(5)) - 5; i < num_elements; ++i)
//     {
//         std::cout << host_data[i] << " ";
//     }
//     std::cout << std::endl;
// }

#endif // USE_HIP
