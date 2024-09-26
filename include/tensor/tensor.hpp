// tensor.hpp
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>

#include "device/device.hpp"
#include "tensor/buffer.hpp"

class Tensor
{
public:
    Tensor(DeviceType device_type = DeviceType::CPU, size_t device_id = 0);
    Tensor(TensorDataType dtype, const std::vector<size_t> &dims, DeviceType device_type = DeviceType::CPU, size_t device_id = 0);

    const std::vector<size_t> &getDims() const;
    const std::vector<size_t> &getStrides() const;
    size_t getNDim() const;
    size_t getNumElements() const;

    void reshape(const std::vector<size_t> &new_dims);

    void setDataType(TensorDataType dtype);
    TensorDataType getDataType() const;

    void allocateBuffer(TensorDataType dtype, size_t num_elements);
    std::shared_ptr<Buffer> getBuffer();
    std::shared_ptr<const Buffer> getBuffer() const;

    template <typename T>
    T *data();

    template <typename T>
    const T *data() const;

    template <typename T>
    void setData(const std::vector<T> &data);

    void freeData();

    size_t getLinearIndex(const std::vector<int64_t> &indices) const;
    std::string toString() const;

    // get the tuple of device type and device index
    std::pair<DeviceType, size_t> getDevice() const;

private:
    // 
    DeviceType device_type_;
    size_t device_id_;

    TensorDataType data_type_;
    std::vector<size_t> dimensions_;
    std::vector<size_t> strides_;
    size_t num_elements_;

    std::shared_ptr<Buffer> buffer_;

    // Helper methods
    std::vector<size_t> calcStrides(const std::vector<size_t> &dims);
    size_t calcNumElements(const std::vector<size_t> &dims);
};

Tensor create_tensor(TensorDataType dtype, const std::vector<size_t> &dims, const std::vector<float> &data, DeviceType device_type = DeviceType::CPU, size_t device_id = 0);

#endif // TENSOR_HPP
