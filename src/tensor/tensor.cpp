// tensor.cpp
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include "tensor/tensor.hpp"
#include "tensor/buffer.hpp"
#include "tensor/tensor_utils.hpp"
#include "utils.hpp"

Tensor::Tensor(Device *device)
    : data_type_(TensorDataType::UNDEFINED), num_elements_(0), buffer_(nullptr), device_(device)
{
}

Tensor::Tensor(TensorDataType dtype, const std::vector<size_t> &dims, Device *device)
    : data_type_(dtype), num_elements_(calcNumElements(dims)), device_(device)
{
    buffer_ = Buffer::create(device, dtype, num_elements_);

    dimensions_ = dims;
#ifdef USE_HIP
    if (device_->getType() == DeviceType::HIP)
    {
        hipErrorCheck(hipMalloc(&d_dimensions_, dims.size() * sizeof(size_t)));
        hipErrorCheck(hipMemcpy(d_dimensions_, dims.data(), dims.size() * sizeof(size_t), hipMemcpyHostToDevice));
    }
#endif

    calculateAndSetStrides(dims);
}

// template <typename T>
// Tensor::Tensor(TensorDataType dtype, const std::vector<size_t> &dims, const std::vector<T> &data, Device *device)
//     : data_type_(dtype), num_elements_(calcNumElements(dims)), device_(device)
// {
//     if (data.size() != num_elements_)
//     {
//         throw std::invalid_argument("Data size does not match tensor dimensions.");
//     }

//     // Allocate buffer for the tensor
//     buffer_ = Buffer::create(device, dtype, num_elements_);

//     // Set the data into the buffer
//     buffer_->setData(data);

//     // Set the dimensions and calculate the strides
//     dimensions_ = dims;
// #ifdef USE_HIP
//     if (device_->getType() == DeviceType::HIP)
//     {
//         // Allocate device memory for dimensions and copy from host
//         hipErrorCheck(hipMalloc(&d_dimensions_, dims.size() * sizeof(size_t)));
//         hipErrorCheck(hipMemcpy(d_dimensions_, dims.data(), dims.size() * sizeof(size_t), hipMemcpyHostToDevice));
//     }
// #endif
//     calculateAndSetStrides(dims);
// }

size_t Tensor::calcNumElements(const std::vector<size_t> &dims)
{
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

void Tensor::calculateAndSetStrides(const std::vector<size_t> &dims)
{
    std::vector<size_t> stride(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i)
    {
        stride[i] = stride[i + 1] * dims[i + 1];
    }
    strides_ = stride;

#ifdef USE_HIP
    if (device_->getType() == DeviceType::HIP)
    {
        hipErrorCheck(hipMalloc(&d_strides_, dims.size() * sizeof(size_t)));
        hipErrorCheck(hipMemcpy(d_strides_, strides_.data(), dims.size() * sizeof(size_t), hipMemcpyHostToDevice));
    }
#endif
}

std::vector<size_t> Tensor::getDims() const
{
    return dimensions_;
}

#ifdef USE_HIP
size_t *Tensor::d_getDims() const
{
    return d_dimensions_;
}

size_t *Tensor::d_getStrides() const
{
    return d_strides_;
}
#endif

std::vector<size_t> Tensor::getStrides() const
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
        std::cerr << "New number of elements does not match the old number of elements." << std::endl;
        std::cerr << "Old number of elements: " << num_elements_ << std::endl;
        std::cerr << "New number of elements: " << new_num_elements << std::endl;
        throw std::runtime_error("Number of elements mismatch.");
    }
    dimensions_ = new_dims;
    calculateAndSetStrides(new_dims);
}

void Tensor::setDataType(TensorDataType dtype)
{
    data_type_ = dtype;
    buffer_ = Buffer::create(device_, dtype, num_elements_);
}

void Tensor::copyFrom(const Tensor &src)
{
    // should work if the number of elements is the same
    if (num_elements_ != src.num_elements_)
    {
        throw std::runtime_error("Number of elements mismatch.");
    }

    if (data_type_ != src.data_type_)
    {
        throw std::runtime_error("Data type mismatch.");
    }

    buffer_->copyFrom(src.buffer_.get());
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
    // call setData on the buffer
    buffer_->setData(data);
}

void Tensor::freeData()
{
    buffer_.reset(); // is it necessary?
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

    // Print dimensions
    for (size_t i = 0; i < dimensions_.size(); ++i)
    {
        oss << dimensions_[i];
        if (i < dimensions_.size() - 1)
        {
            oss << ", ";
        }
    }
    oss << "], data=[";

    // Helper lambda to handle data printing
    auto printData = [&](auto *data_ptr)
    {
        size_t num_printed = 0;
        for (size_t i = 0; i < num_elements_; ++i)
        {
            oss << data_ptr[i];
            if (i < num_elements_ - 1)
            {
                oss << ", ";
            }
            if (++num_printed > 5)
            {
                oss << "...";
                break;
            }
        }
    };

    size_t num_printed = 0;

    switch (data_type_)
    {
    case TensorDataType::FLOAT32:
        printData(data<float>());
        break;

    case TensorDataType::FLOAT64:
        printData(data<double>());
        break;

    case TensorDataType::INT32:
        printData(data<int32_t>());
        break;

    case TensorDataType::INT64:
        printData(data<int64_t>());
        break;

    case TensorDataType::INT8:
        printData(reinterpret_cast<const int *>(data<int8_t>()));
        break;

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
        buffer_ = Buffer::create(device_, dtype, num_elements);
    }
    else
    {
        buffer_->resize(num_elements);
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

Tensor create_tensor(TensorDataType dtype, const std::vector<size_t> &dims, const std::vector<float> &data, Device *device)
{
    size_t num_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

    if (data.size() != num_elements)
    {
        throw std::invalid_argument("Data size does not match tensor dimensions.");
    }

    Tensor tensor(dtype, dims, device);

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

void Tensor::to(Device *device)
{
    if (device->getType() != device_->getType())
    {
        Buffer *new_buffer = buffer_->to(device);
        buffer_ = std::shared_ptr<Buffer>(new_buffer);
        device_ = device;
    }
}

// void *getDataPointer() override;
// const void *getDataPointer() const override;

void *Tensor::getDataPointer()
{
    return buffer_->getDataPointer();
}

const void *Tensor::getDataPointer() const
{
    return buffer_->getDataPointer();
}