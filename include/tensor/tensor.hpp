#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <array>
#include <variant>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <memory>
#include <vector>

#include "device/device.hpp"
#include "tensor/buffer.hpp"

enum class TensorDataType
{
    UNDEFINED,
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    INT8,
    UINT8,
    
    /// XXX: Should "half" included in the CPU backend? AVX512 can use float16, but what about non-AVX512 implementations?
    FLOAT16
};

class Tensor
{
public:
    Tensor();
    Tensor(TensorDataType dtype, const std::vector<size_t> &dims);

    const std::vector<size_t> &getDims() const;
    const std::vector<size_t> &getStrides() const;
    size_t getNDim() const;
    size_t getNumElements() const;

    void copy_tensor(const Tensor &other);
    void reshape(const std::vector<size_t> &new_dims);

    void setDataType(TensorDataType);
    TensorDataType getDataType() const;

    template <typename T>
    T *data();

    template <typename T>
    const T *data() const;

    template <typename T>
    void setData(T *data, size_t size);

    template <typename T>
    void setDataPointer(T *data, const std::vector<size_t> &dims);

    void freeData();

    size_t getLinearIndex(const std::vector<int64_t> &indices) const;

    // to string
    std::string toString() const;

private:
    TensorDataType data_type;
    std::vector<size_t> dimensions;
    std::vector<size_t> strides;
    size_t num_elements;

    std::variant<
        std::monostate,
        float *,
        double *,
        int32_t *,
        int64_t *,
        int8_t *,
        uint8_t *>
        values;

    std::vector<size_t> calcStrides(const std::vector<size_t> &dims);
    size_t calcNumElements(const std::vector<size_t> &dims);
    void allocateData(TensorDataType dtype);
};

Tensor create_tensor(TensorDataType dtype, const std::vector<size_t> &dims, const std::vector<float> &data);

namespace TensorUtils
{
    enum class TensorCompareResult
    {
        EQUAL,
        SHAPE_MISMATCH,
        DATA_TYPE_MISMATCH,
        DATA_MISMATCH,
    };

    std::string TensorCompareResultToString(TensorCompareResult result);

    size_t getDataTypeSize(TensorDataType dtype);
    std::string getDataTypeName(TensorDataType dtype);
    TensorCompareResult areTensorsEqual(const Tensor &lhs, const Tensor &rhs);
} // namespace TensorUtils

#endif // TENSOR_HPP
