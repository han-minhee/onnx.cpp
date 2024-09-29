// tensor_utils.hpp
#ifndef TENSOR_UTILS_HPP
#define TENSOR_UTILS_HPP

#include <string>

#include "enums.hpp"
#include "types/half_t.hpp"
#include "tensor/tensor.hpp"

namespace TensorUtils
{
    size_t getDataTypeSize(TensorDataType dtype);
    std::string getDataTypeName(TensorDataType dtype);

    TensorCompareResult areTensorsEqual(const Tensor &A, const Tensor &B);

    std::string TensorCompareResultToString(TensorCompareResult result);

    template <typename T>
    TensorDataType getDataTypeFromType();

#define DEFINE_CPP_TYPE_TO_TENSOR_TYPE(cpp_type, tensor_type) \
    template <>                                               \
    inline TensorDataType getDataTypeFromType<cpp_type>()     \
    {                                                         \
        return tensor_type;                                   \
    }

    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(void, TensorDataType::UNDEFINED)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(float, TensorDataType::FLOAT32)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(double, TensorDataType::FLOAT64)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(int32_t, TensorDataType::INT32)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(int64_t, TensorDataType::INT64)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(int8_t, TensorDataType::INT8)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(uint8_t, TensorDataType::UINT8)

    // Custom half type
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(half_t, TensorDataType::FLOAT16)

#undef DEFINE_CPP_TYPE_TO_TENSOR_TYPE
}

#endif // TENSOR_UTILS_HPP
