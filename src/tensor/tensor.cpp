#include "tensor/tensor.hpp"
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <sstream>

namespace TensorUtils
{
    size_t getDataTypeSize(TensorDataType dtype)
    {
        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            return sizeof(float);
        case TensorDataType::FLOAT64:
            return sizeof(double);
        case TensorDataType::INT32:
            return sizeof(int32_t);
        case TensorDataType::INT64:
            return sizeof(int64_t);
        case TensorDataType::INT8:
            return sizeof(int8_t);
        case TensorDataType::UINT8:
            return sizeof(uint8_t);
        case TensorDataType::UNDEFINED:
            return 0;
        default:
            throw std::runtime_error("Unsupported data type in getDataTypeSize");
        }
    }
    std::string getDataTypeName(TensorDataType dtype)
    {
        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            return "FLOAT32";
        case TensorDataType::FLOAT64:
            return "FLOAT64";
        case TensorDataType::INT32:
            return "INT32";
        case TensorDataType::INT64:
            return "INT64";
        case TensorDataType::INT8:
            return "INT8";
        case TensorDataType::UINT8:
            return "UINT8";
        case TensorDataType::UNDEFINED:
            return "UNDEFINED";
        default:
            throw std::runtime_error("Unsupported data type in getDataTypeName");
        }
    }

    TensorCompareResult areTensorsEqual(const Tensor &lhs, const Tensor &rhs)
    {
        // check if the data types are the same
        if (lhs.getDataType() != rhs.getDataType())
        {
            return TensorCompareResult::DATA_TYPE_MISMATCH;
        }

        // check if the dimensions are the same
        if (lhs.getDims() != rhs.getDims())
        {
            return TensorCompareResult::SHAPE_MISMATCH;
        }

        // check if the data is no more different than the tolerance
        // tolerance is get by getting the largest of the absolute values
        // and 1% of it.

        switch (lhs.getDataType())
        {
        case TensorDataType::FLOAT32:
        {
            const float *lhs_data = lhs.data<float>();
            const float *rhs_data = rhs.data<float>();

            // get the largest of the absolute values
            float max_val = 0.0f;
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                float abs_val = std::abs(lhs_data[i]);
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                }
            }
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                float abs_val = std::abs(rhs_data[i]);
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                }
            }

            float tolerance = max_val * 1e-3;

            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (std::abs(lhs_data[i] - rhs_data[i]) > tolerance)
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        case TensorDataType::FLOAT64:
        {
            const double *lhs_data = lhs.data<double>();
            const double *rhs_data = rhs.data<double>();

            // get the largest of the absolute values
            double max_val = 0.0;
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                double abs_val = std::abs(lhs_data[i]);
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                }
            }
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                double abs_val = std::abs(rhs_data[i]);
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                }
            }

            double tolerance = max_val * 1e-3;

            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (std::abs(lhs_data[i] - rhs_data[i]) > tolerance)
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        case TensorDataType::INT32:
        {
            const int32_t *lhs_data = lhs.data<int32_t>();
            const int32_t *rhs_data = rhs.data<int32_t>();
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (lhs_data[i] != rhs_data[i])
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        case TensorDataType::INT64:
        {
            const int64_t *lhs_data = lhs.data<int64_t>();
            const int64_t *rhs_data = rhs.data<int64_t>();
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (lhs_data[i] != rhs_data[i])
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        case TensorDataType::INT8:
        {
            const int8_t *lhs_data = lhs.data<int8_t>();
            const int8_t *rhs_data = rhs.data<int8_t>();
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (lhs_data[i] != rhs_data[i])
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        case TensorDataType::UINT8:
        {
            const uint8_t *lhs_data = lhs.data<uint8_t>();
            const uint8_t *rhs_data = rhs.data<uint8_t>();
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (lhs_data[i] != rhs_data[i])
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        default:
            throw std::runtime_error("Unsupported data type in areTensorsEqual");
        }
        return TensorCompareResult::EQUAL;
    }

    std::string TensorCompareResultToString(TensorCompareResult result)
    {
        switch (result)
        {
        case TensorCompareResult::EQUAL:
            return "EQUAL";
        case TensorCompareResult::SHAPE_MISMATCH:
            return "SHAPE_MISMATCH";
        case TensorCompareResult::DATA_TYPE_MISMATCH:
            return "DATA_TYPE_MISMATCH";
        case TensorCompareResult::DATA_MISMATCH:
            return "DATA_MISMATCH";
        default:
            throw std::runtime_error("Unsupported TensorCompareResult in TensorCompareResultToString");
        }
    }
}

Tensor::Tensor() : data_type(TensorDataType::UNDEFINED), num_elements(0), values(std::monostate{}) {}

Tensor::Tensor(TensorDataType dtype, const std::vector<size_t> &dims)
    : data_type(dtype), dimensions(dims), num_elements(calcNumElements(dims))
{
    // std::cout << "Tensor constructor called" << std::endl;
    strides = calcStrides(dims);
    // std::cout << "Allocating data" << std::endl;
    allocateData(dtype);
    // std::cout << "Data allocated" << std::endl;
}

size_t Tensor::calcNumElements(const std::vector<size_t> &dims)
{
    // std::cout << "Calculating number of elements" << std::endl;
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

std::vector<size_t> Tensor::calcStrides(const std::vector<size_t> &dims)
{
    std::vector<size_t> stride(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i)
    {
        stride[i] = stride[i + 1] * dims[i + 1];
    }
    return stride;
}

void Tensor::copy_tensor(const Tensor &other)
{
    data_type = other.data_type;
    dimensions = other.dimensions;
    strides = other.strides;
    num_elements = other.num_elements;

    switch (data_type)
    {
    case TensorDataType::FLOAT32:
        values = new float[num_elements];
        std::copy(other.data<float>(), other.data<float>() + num_elements, data<float>());
        break;
    case TensorDataType::FLOAT64:
        values = new double[num_elements];
        std::copy(other.data<double>(), other.data<double>() + num_elements, data<double>());
        break;
    case TensorDataType::INT32:
        values = new int32_t[num_elements];
        std::copy(other.data<int32_t>(), other.data<int32_t>() + num_elements, data<int32_t>());
        break;
    case TensorDataType::INT64:
        values = new int64_t[num_elements];
        std::copy(other.data<int64_t>(), other.data<int64_t>() + num_elements, data<int64_t>());
        break;
    case TensorDataType::INT8:
        values = new int8_t[num_elements];
        std::copy(other.data<int8_t>(), other.data<int8_t>() + num_elements, data<int8_t>());
        break;
    case TensorDataType::UINT8:
        values = new uint8_t[num_elements];
        std::copy(other.data<uint8_t>(), other.data<uint8_t>() + num_elements, data<uint8_t>());
        break;
    default:
        throw std::runtime_error("Unsupported data type in copy_tensor");
    }
}

void Tensor::allocateData(TensorDataType dtype)
{
    if (num_elements == 0)
    {
        return;
    }

    switch (dtype)
    {
    case TensorDataType::FLOAT32:
        values = new float[num_elements];
        break;
    case TensorDataType::FLOAT64:
        values = new double[num_elements];
        break;
    case TensorDataType::INT32:
        values = new int32_t[num_elements];
        break;
    case TensorDataType::INT64:
        values = new int64_t[num_elements];
        break;
    case TensorDataType::INT8:
        values = new int8_t[num_elements];
        break;
    case TensorDataType::UINT8:
        values = new uint8_t[num_elements];
        break;
    default:
        std::cout << "Unsupported data type in allocateDate: " << TensorUtils::getDataTypeName(dtype) << std::endl;
        throw std::runtime_error("Unsupported data type in allocateData");
    }
}

std::string Tensor::toString() const
{
    std::ostringstream oss;
    // get the dtype, dims, and first 5 elements
    oss << "Tensor: dtype=" << TensorUtils::getDataTypeName(data_type) << ", dims=[";
    for (size_t i = 0; i < dimensions.size(); ++i)
    {
        oss << dimensions[i];
        if (i < dimensions.size() - 1)
        {
            oss << ", ";
        }
    }
    oss << "], data=[";
    size_t num_printed = 0;
    switch (data_type)
    {
    case TensorDataType::FLOAT32:
    {
        const float *data = this->data<float>();
        for (size_t i = 0; i < num_elements; ++i)
        {
            oss << data[i];
            if (i < num_elements - 1)
            {
                oss << ", ";
            }
            if (num_printed++ >= 5)
            {
                oss << "...";
                break;
            }
        }
        break;
    }
    case TensorDataType::FLOAT64:
    {
        const double *data = this->data<double>();
        for (size_t i = 0; i < num_elements; ++i)
        {
            oss << data[i];
            if (i < num_elements - 1)
            {
                oss << ", ";
            }
            if (num_printed++ >= 5)
            {
                oss << "...";
                break;
            }
        }
        break;
    }

    case TensorDataType::INT32:
    {
        const int32_t *data = this->data<int32_t>();
        for (size_t i = 0; i < num_elements; ++i)
        {
            oss << data[i];
            if (i < num_elements - 1)
            {
                oss << ", ";
            }
            if (num_printed++ >= 5)
            {
                oss << "...";
                break;
            }
        }
        break;
    }

    case TensorDataType::INT64:
    {
        const int64_t *data = this->data<int64_t>();
        for (size_t i = 0; i < num_elements; ++i)
        {
            oss << data[i];
            if (i < num_elements - 1)
            {
                oss << ", ";
            }
            if (num_printed++ >= 5)
            {
                oss << "...";
                break;
            }
        }
        break;
    }
    deafult:
        oss << "Unsupported data type";
    }

    oss << "]";

    return oss.str();
}

void Tensor::reshape(const std::vector<size_t> &new_dims)
{
    size_t new_num_elements = calcNumElements(new_dims);
    if (num_elements > 0 && new_num_elements != num_elements)
    {
        throw std::runtime_error("New shape must have the same number of elements unless the tensor is empty.");
    }
    dimensions = new_dims;
    strides = calcStrides(new_dims);
    num_elements = new_num_elements;
}

void Tensor::freeData()
{

    if (std::holds_alternative<float *>(values))
        delete[] std::get<float *>(values);
    else if (std::holds_alternative<double *>(values))
        delete[] std::get<double *>(values);
    else if (std::holds_alternative<int32_t *>(values))
        delete[] std::get<int32_t *>(values);
    else if (std::holds_alternative<int64_t *>(values))
        delete[] std::get<int64_t *>(values);
    else if (std::holds_alternative<int8_t *>(values))
        delete[] std::get<int8_t *>(values);
    else if (std::holds_alternative<uint8_t *>(values))
        delete[] std::get<uint8_t *>(values);

    values = std::monostate{};
    num_elements = 0;
    dimensions.clear();
    strides.clear();
}

template <typename T>
void Tensor::setData(T *data, size_t size)
{
    if constexpr (std::is_same<T, float>::value)
    {
        values = new float[size];
        std::copy(data, data + size, std::get<float *>(values));
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        values = new double[size];
        std::copy(data, data + size, std::get<double *>(values));
    }
    else if constexpr (std::is_same<T, int32_t>::value)
    {
        values = new int32_t[size];
        std::copy(data, data + size, std::get<int32_t *>(values));
    }
    else if constexpr (std::is_same<T, int64_t>::value)
    {
        values = new int64_t[size];
        std::copy(data, data + size, std::get<int64_t *>(values));
    }
    else if constexpr (std::is_same<T, int8_t>::value)
    {
        values = new int8_t[size];
        std::copy(data, data + size, std::get<int8_t *>(values));
    }
    else if constexpr (std::is_same<T, uint8_t>::value)
    {
        values = new uint8_t[size];
        std::copy(data, data + size, std::get<uint8_t *>(values));
    }
    else
    {
        throw std::runtime_error("Unsupported data type in setData");
    }
}

template <typename T>
void Tensor::setDataPointer(T *data, const std::vector<size_t> &dims)
{

    freeData();

    if constexpr (std::is_same<T, float>::value)
    {
        data_type = TensorDataType::FLOAT32;
        values = data;
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        data_type = TensorDataType::FLOAT64;
        values = data;
    }
    else if constexpr (std::is_same<T, int32_t>::value)
    {
        data_type = TensorDataType::INT32;
        values = data;
    }
    else if constexpr (std::is_same<T, int64_t>::value)
    {
        data_type = TensorDataType::INT64;
        values = data;
    }
    else if constexpr (std::is_same<T, int8_t>::value)
    {
        data_type = TensorDataType::INT8;
        values = data;
    }
    else if constexpr (std::is_same<T, uint8_t>::value)
    {
        data_type = TensorDataType::UINT8;
        values = data;
    }
    else
    {
        throw std::runtime_error("Unsupported data type in setDataPointer.");
    }

    dimensions = dims;
    num_elements = calcNumElements(dims);
    strides = calcStrides(dims);
}

template <typename T>
T *Tensor::data()
{
    if (!std::holds_alternative<T *>(values))
    {
        throw std::runtime_error("Incorrect data type access");
    }
    return std::get<T *>(values);
}

template <typename T>
const T *Tensor::data() const
{
    if (!std::holds_alternative<T *>(values))
    {
        throw std::runtime_error("Incorrect data type access");
    }
    return std::get<T *>(values);
}

const std::vector<size_t> &Tensor::getDims() const
{
    return dimensions;
}

const std::vector<size_t> &Tensor::getStrides() const
{
    return strides;
}

size_t Tensor::getNDim() const
{
    return dimensions.size();
}

size_t Tensor::getNumElements() const
{
    return num_elements;
}

void Tensor::setDataType(TensorDataType dtype)
{
    data_type = dtype;
}

TensorDataType Tensor::getDataType() const
{
    return data_type;
}

Tensor create_tensor(TensorDataType dtype, const std::vector<size_t> &dims, const std::vector<float> &data)
{

    size_t num_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

    if (data.size() != num_elements)
    {
        throw std::invalid_argument("Data size does not match tensor dimensions.");
    }

    Tensor tensor(dtype, dims);

    switch (dtype)
    {
    case TensorDataType::FLOAT32:
    {

        float *tensor_data = tensor.data<float>();
        std::copy(data.begin(), data.end(), tensor_data);
        break;
    }
    case TensorDataType::FLOAT64:
    {

        double *tensor_data = tensor.data<double>();
        std::transform(data.begin(), data.end(), tensor_data, [](float val) -> double
                       { return static_cast<double>(val); });
        break;
    }
    case TensorDataType::INT32:
    {

        int32_t *tensor_data = tensor.data<int32_t>();
        std::transform(data.begin(), data.end(), tensor_data, [](float val) -> int32_t
                       { return static_cast<int32_t>(val); });
        break;
    }
    case TensorDataType::INT64:
    {

        int64_t *tensor_data = tensor.data<int64_t>();
        std::transform(data.begin(), data.end(), tensor_data, [](float val) -> int64_t
                       { return static_cast<int64_t>(val); });
        break;
    }
    case TensorDataType::INT8:
    {

        int8_t *tensor_data = tensor.data<int8_t>();
        std::transform(data.begin(), data.end(), tensor_data, [](float val) -> int8_t
                       {
                if(val < static_cast<float>(std::numeric_limits<int8_t>::min())) return std::numeric_limits<int8_t>::min();
                if(val > static_cast<float>(std::numeric_limits<int8_t>::max())) return std::numeric_limits<int8_t>::max();
                return static_cast<int8_t>(val); });
        break;
    }
    case TensorDataType::UINT8:
    {

        uint8_t *tensor_data = tensor.data<uint8_t>();
        std::transform(data.begin(), data.end(), tensor_data, [](float val) -> uint8_t
                       {
                if(val < 0.0f) return 0;
                if(val > static_cast<float>(std::numeric_limits<uint8_t>::max())) return std::numeric_limits<uint8_t>::max();
                return static_cast<uint8_t>(val); });
        break;
    }
    default:
        throw std::invalid_argument("Unsupported TensorDataType for create_tensor.");
    }

    return tensor;
}

size_t Tensor::getLinearIndex(const std::vector<int64_t> &indices) const
{
    if (indices.size() != dimensions.size())
    {
        throw std::runtime_error("Index dimension mismatch.");
    }

    size_t linear_index = 0;
    for (size_t i = 0; i < dimensions.size(); ++i)
    {
        if (indices[i] < 0 || indices[i] >= static_cast<int64_t>(dimensions[i]))
        {
            throw std::runtime_error("Index out of bounds.");
        }
        linear_index += indices[i] * strides[i];
    }
    return linear_index;
}

#define INSTANTIATE_TENSOR_TEMPLATE(T)                       \
    template T *Tensor::data<T>();                           \
    template const T *Tensor::data<T>() const;               \
    template void Tensor::setData<T>(T * data, size_t size); \
    template void Tensor::setDataPointer<T>(T * data, const std::vector<size_t> &dims);

INSTANTIATE_TENSOR_TEMPLATE(float)
INSTANTIATE_TENSOR_TEMPLATE(double)
INSTANTIATE_TENSOR_TEMPLATE(int32_t)
INSTANTIATE_TENSOR_TEMPLATE(int64_t)
INSTANTIATE_TENSOR_TEMPLATE(int8_t)
INSTANTIATE_TENSOR_TEMPLATE(uint8_t)

#undef INSTANTIATE_TENSOR_TEMPLATE