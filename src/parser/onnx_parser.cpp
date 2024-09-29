#include <iostream>
#include <fstream>
#include <variant>

#include "parser/onnx_parser.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"

#include "tensor/tensor_utils.hpp"

using AttributeValue = std::variant<int, float, std::vector<int64_t>, std::vector<float>, Tensor>;

TensorDataType convertONNXDataType(int onnx_data_type)
{
    switch (onnx_data_type)
    {
    case onnx::TensorProto_DataType_FLOAT:
        return TensorDataType::FLOAT32;
    case onnx::TensorProto_DataType_DOUBLE:
        return TensorDataType::FLOAT64;
    case onnx::TensorProto_DataType_INT32:
        return TensorDataType::INT32;
    case onnx::TensorProto_DataType_INT64:
        return TensorDataType::INT64;
    case onnx::TensorProto_DataType_INT8:
        return TensorDataType::INT8;
    case onnx::TensorProto_DataType_UINT8:
        return TensorDataType::UINT8;
    case onnx::TensorProto_DataType_FLOAT16:
        return TensorDataType::FLOAT16;
    default:
        return TensorDataType::UNDEFINED;
    }
}

/// XXX: The CPU Device should be given for this
Tensor parseONNXTensor(const onnx::TensorProto &onnx_tensor, Device *device)
{
    TensorDataType data_type = convertONNXDataType(onnx_tensor.data_type());

    // Get the dimensions
    std::vector<size_t> dims;
    for (int i = 0; i < onnx_tensor.dims_size(); ++i)
    {
        dims.push_back(onnx_tensor.dims(i));
    }

    // Create a Tensor object
    Tensor tensor(data_type, dims, device);

    // Calculate the number of elements
    size_t num_elements = tensor.getNumElements();

    // Check if raw_data is present
    if (!onnx_tensor.raw_data().empty())
    {
        // Get a pointer to the raw data
        const std::string &raw_data = onnx_tensor.raw_data();

        // Ensure that the size of raw_data matches the expected size
        size_t expected_size = num_elements * TensorUtils::getDataTypeSize(data_type);
        if (raw_data.size() != expected_size)
        {
            throw std::runtime_error("Size of raw_data does not match expected size.");
        }

        // Allocate buffer for the tensor
        tensor.allocateBuffer(data_type, num_elements);

        // Copy data directly into the tensor's buffer
        switch (data_type)
        {
        case TensorDataType::FLOAT32:
            std::memcpy(tensor.data<float>(), raw_data.data(), expected_size);
            break;
        case TensorDataType::FLOAT64:
            std::memcpy(tensor.data<double>(), raw_data.data(), expected_size);
            break;
        case TensorDataType::INT32:
            std::memcpy(tensor.data<int32_t>(), raw_data.data(), expected_size);
            break;
        case TensorDataType::INT64:
            std::memcpy(tensor.data<int64_t>(), raw_data.data(), expected_size);
            break;
        case TensorDataType::INT8:
            std::memcpy(tensor.data<int8_t>(), raw_data.data(), expected_size);
            break;
        case TensorDataType::UINT8:
            std::memcpy(tensor.data<uint8_t>(), raw_data.data(), expected_size);
            break;
        case TensorDataType::FLOAT16:
            std::memcpy(tensor.data<half_t>(), raw_data.data(), expected_size);
            break;
        default:
            throw std::runtime_error("Unsupported data type.");
        }
    }
    else
    {
        // Handle the case where data is stored in repeated fields
        tensor.allocateBuffer(data_type, num_elements);

        switch (data_type)
        {
        case TensorDataType::FLOAT32:
        {
            size_t num_elements = onnx_tensor.float_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for FLOAT32 data.");
            }
            std::copy(onnx_tensor.float_data().begin(), onnx_tensor.float_data().end(), tensor.data<float>());
            break;
        }
        case TensorDataType::FLOAT64:
        {
            size_t num_elements = onnx_tensor.double_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for FLOAT64 data.");
            }
            std::copy(onnx_tensor.double_data().begin(), onnx_tensor.double_data().end(), tensor.data<double>());
            break;
        }
        case TensorDataType::INT32:
        {
            size_t num_elements = onnx_tensor.int32_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for INT32 data.");
            }
            std::copy(onnx_tensor.int32_data().begin(), onnx_tensor.int32_data().end(), tensor.data<int32_t>());
            break;
        }
        case TensorDataType::INT64:
        {
            size_t num_elements = onnx_tensor.int64_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for INT64 data.");
            }
            std::copy(onnx_tensor.int64_data().begin(), onnx_tensor.int64_data().end(), tensor.data<int64_t>());
            break;
        }
        case TensorDataType::INT8:
        {
            size_t num_elements = onnx_tensor.int32_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for INT8 data.");
            }
            int8_t *data_ptr = tensor.data<int8_t>();
            std::transform(onnx_tensor.int32_data().begin(), onnx_tensor.int32_data().end(), data_ptr, [](int32_t val)
                           { return static_cast<int8_t>(val); });
            break;
        }
        case TensorDataType::UINT8:
        {
            size_t num_elements = onnx_tensor.int32_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for UINT8 data.");
            }
            uint8_t *data_ptr = tensor.data<uint8_t>();
            std::transform(onnx_tensor.int32_data().begin(), onnx_tensor.int32_data().end(), data_ptr, [](int32_t val)
                           { return static_cast<uint8_t>(val); });
            break;
        }

        case TensorDataType::FLOAT16:
        {
            size_t num_elements = onnx_tensor.int32_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for FLOAT16 data.");
            }
            half_t *data_ptr = tensor.data<half_t>();
            std::transform(onnx_tensor.int32_data().begin(), onnx_tensor.int32_data().end(), data_ptr, [](int32_t val)
                           { return static_cast<half_t>(val); });
            break;
        }

        default:
            throw std::runtime_error("Unsupported data type in type-specific fields.");
        }
    }

    return tensor;
}
