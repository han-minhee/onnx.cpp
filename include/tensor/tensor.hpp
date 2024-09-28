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
    Tensor(Device *device = new CpuDevice());
    Tensor(TensorDataType dtype, const std::vector<size_t> &dims, Device *device = new CpuDevice());

    std::vector<size_t> getDims() const;
    std::vector<size_t> getStrides() const;

#ifdef USE_HIP
    size_t *getDimsPointer();
    size_t *getStridesPointer();
#endif

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

    void to(Device *device);
    Device *getDevice();

    void *getDataPointer();
    const void *getDataPointer() const;

    void copyFrom(const Tensor &src);

#ifdef USE_HIP
    size_t *d_getDims() const;
    size_t *d_getStrides() const;
#endif

private:
    Device *device_;
    TensorDataType data_type_;
    std::vector<size_t> dimensions_;
    std::vector<size_t> strides_;

#ifdef USE_HIP
    size_t *d_dimensions_;
    size_t *d_strides_;

#endif

    size_t num_elements_;

    std::shared_ptr<Buffer> buffer_;

    void calculateAndSetStrides(const std::vector<size_t> &dims);
    size_t calcNumElements(const std::vector<size_t> &dims);
};

Tensor create_tensor(TensorDataType dtype, const std::vector<size_t> &dims, const std::vector<float> &data, Device *device = new CpuDevice());

#endif // TENSOR_HPP
