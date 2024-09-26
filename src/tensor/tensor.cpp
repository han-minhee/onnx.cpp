// tensor.cpp
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include "tensor/tensor.hpp"
#include "tensor/buffer.hpp"
#include "tensor/tensor_utils.hpp"

Tensor::Tensor()
    : data_type_(TensorDataType::UNDEFINED), num_elements_(0), buffer_(nullptr)
{
}

Tensor::Tensor(TensorDataType dtype, const std::vector<size_t> &dims)
    : data_type_(dtype), dimensions_(dims), num_elements_(calcNumElements(dims))
{
    strides_ = calcStrides(dims);
    buffer_ = std::make_shared<CpuBuffer>(dtype, num_elements_);
}

size_t Tensor::calcNumElements(const std::vector<size_t> &dims)
{
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

const std::vector<size_t> &Tensor::getDims() const
{
    return dimensions_;
}

const std::vector<size_t> &Tensor::getStrides() const
{
    return strides_;
}


std::shared_ptr<Buffer> Tensor::getBuffer()
{
    return buffer_;
}

std::shared_ptr<const Buffer> Tensor::getBuffer() const
{
    return buffer_;
}

size_t Tensor::getNDim() const
{
    return dimensions_.size();
}

size_t Tensor::getNumElements() const
{
    return num_elements_;
}

void Tensor::reshape(const std::vector<size_t> &new_dims)
{
    size_t new_num_elements = calcNumElements(new_dims);
    if (new_num_elements != num_elements_)
    {
        std:: cerr << "New number of elements does not match the old number of elements." << std::endl;
        std:: cerr << "Old number of elements: " << num_elements_ << std::endl;
        std:: cerr << "New number of elements: " << new_num_elements << std::endl;
        throw std::runtime_error("Number of elements mismatch.");
    }
    dimensions_ = new_dims;
    strides_ = calcStrides(new_dims);
}

void Tensor::setDataType(TensorDataType dtype)
{
    data_type_ = dtype;
    // Reallocate buffer with new data type
    buffer_ = std::make_shared<CpuBuffer>(dtype, num_elements_);
}

TensorDataType Tensor::getDataType() const
{
    return data_type_;
}

template <typename T>
T *Tensor::data()
{
    if (TensorUtils::getDataTypeFromType<T>() != data_type_)
    {
        throw std::runtime_error("Incorrect data type access");
    }
    return static_cast<T *>(buffer_->getDataPointer());
}

template <typename T>
const T *Tensor::data() const
{
    if (TensorUtils::getDataTypeFromType<T>() != data_type_)
    {
        throw std::runtime_error("Incorrect data type access");
    }
    return static_cast<const T *>(buffer_->getDataPointer());
}

template <typename T>
void Tensor::setData(const std::vector<T> &data)
{
    if (TensorUtils::getDataTypeFromType<T>() != data_type_)
    {
        throw std::runtime_error("Incorrect data type");
    }
    if (data.size() != num_elements_)
    {
        throw std::runtime_error("Data size mismatch");
    }
    T *buffer_data = this->data<T>();
    std::copy(data.begin(), data.end(), buffer_data);
}

void Tensor::freeData()
{
    buffer_.reset();
    num_elements_ = 0;
    dimensions_.clear();
    strides_.clear();
}

size_t Tensor::getLinearIndex(const std::vector<int64_t> &indices) const
{
    if (indices.size() != dimensions_.size())
    {
        throw std::runtime_error("Index dimension mismatch.");
    }

    size_t linear_index = 0;
    for (size_t i = 0; i < dimensions_.size(); ++i)
    {
        if (indices[i] < 0 || indices[i] >= static_cast<int64_t>(dimensions_[i]))
        {
            throw std::runtime_error("Index out of bounds.");
        }
        linear_index += indices[i] * strides_[i];
    }
    return linear_index;
}

std::string Tensor::toString() const
{
    std::ostringstream oss;
    oss << "Tensor: dtype=" << TensorUtils::getDataTypeName(data_type_) << ", dims=[";
    for (size_t i = 0; i < dimensions_.size(); ++i)
    {
        oss << dimensions_[i];
        if (i < dimensions_.size() - 1)
        {
            oss << ", ";
        }
    }
    oss << "], data=[";
    size_t num_printed = 0;

    switch (data_type_)
    {
    case TensorDataType::FLOAT32:
    {
        const float *data_ptr = data<float>();
        for (size_t i = 0; i < num_elements_; ++i)
        {
            oss << data_ptr[i];
            if (i < num_elements_ - 1)
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
        const double *data_ptr = data<double>();
        for (size_t i = 0; i < num_elements_; ++i)
        {
            oss << data_ptr[i];
            if (i < num_elements_ - 1)
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
        const int32_t *data_ptr = data<int32_t>();
        for (size_t i = 0; i < num_elements_; ++i)
        {
            oss << data_ptr[i];
            if (i < num_elements_ - 1)
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
        const int64_t *data_ptr = data<int64_t>();
        for (size_t i = 0; i < num_elements_; ++i)
        {
            oss << data_ptr[i];
            if (i < num_elements_ - 1)
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

    case TensorDataType::INT8:
    {
        const int8_t *data_ptr = data<int8_t>();
        for (size_t i = 0; i < num_elements_; ++i)
        {
            oss << static_cast<int>(data_ptr[i]);
            if (i < num_elements_ - 1)
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

    default:
        oss << "Unsupported data type";
    }

    oss << "]";
    return oss.str();
}



void Tensor::allocateBuffer(TensorDataType dtype, size_t num_elements)
{
    if (!buffer_ || buffer_->getNumElements() != num_elements)
    {
        buffer_ = std::make_shared<CpuBuffer>(dtype, num_elements);
    }
}

#define INSTANTIATE_TENSOR_TEMPLATE(T)         \
    template T *Tensor::data<T>();             \
    template const T *Tensor::data<T>() const; \
    template void Tensor::setData<T>(const std::vector<T> &data);


// no void type is allowed
INSTANTIATE_TENSOR_TEMPLATE(float)
INSTANTIATE_TENSOR_TEMPLATE(double)
INSTANTIATE_TENSOR_TEMPLATE(int32_t)
INSTANTIATE_TENSOR_TEMPLATE(int64_t)
INSTANTIATE_TENSOR_TEMPLATE(int8_t)
INSTANTIATE_TENSOR_TEMPLATE(uint8_t)

#undef INSTANTIATE_TENSOR_TEMPLATE

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
        tensor.setData<float>(data);
        break;
    case TensorDataType::FLOAT64:
    {
        std::vector<double> data_double(data.begin(), data.end());
        tensor.setData<double>(data_double);
        break;
    }
    case TensorDataType::INT32:
    {
        std::vector<int32_t> data_int32(data.begin(), data.end());
        tensor.setData<int32_t>(data_int32);
        break;
    }

    case TensorDataType::INT64:
    {
        std::vector<int64_t> data_int64(data.begin(), data.end());
        tensor.setData<int64_t>(data_int64);
        break;
    }

    case TensorDataType::INT8:
    {
        std::vector<int8_t> data_int8(data.begin(), data.end());
        tensor.setData<int8_t>(data_int8);
        break;
    }

    case TensorDataType::UINT8:
    {
        std::vector<uint8_t> data_uint8(data.begin(), data.end());
        tensor.setData<uint8_t>(data_uint8);
        break;
    }

    default:
        throw std::invalid_argument("Unsupported TensorDataType for create_tensor.");
    }

    return tensor;
}
