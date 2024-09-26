// tensor_utils.hpp
#ifndef TENSOR_UTILS_HPP
#define TENSOR_UTILS_HPP

#include <string>

#include "enums.hpp"
#include "tensor/tensor.hpp"

#define DEFINE_CPP_TYPE_TO_TENSOR_TYPE(cpp_type, tensor_type) \
    template <>                                               \
    inline TensorDataType getDataTypeFromType<cpp_type>()     \
    {                                                         \
        return tensor_type;                                   \
    }

namespace TensorUtils
{
    size_t getDataTypeSize(TensorDataType dtype);
    std::string getDataTypeName(TensorDataType dtype);

    TensorCompareResult areTensorsEqual(const Tensor &A, const Tensor &B);

    std::string TensorCompareResultToString(TensorCompareResult result);

    template <typename T>
    TensorDataType getDataTypeFromType();

    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(void, TensorDataType::UNDEFINED)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(float, TensorDataType::FLOAT32)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(double, TensorDataType::FLOAT64)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(int32_t, TensorDataType::INT32)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(int64_t, TensorDataType::INT64)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(int8_t, TensorDataType::INT8)
    DEFINE_CPP_TYPE_TO_TENSOR_TYPE(uint8_t, TensorDataType::UINT8)

}

#endif // TENSOR_UTILS_HPP
