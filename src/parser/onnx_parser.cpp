#include <iostream>
#include <fstream>
#include <variant>

#include "parser/onnx_parser.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"

// 4: Unsupported attribute type for: coordinate_transformation_mode
// 4: Unsupported attribute type for: mode
// 4: Unsupported attribute type for: nearest_mode
// 4: Unsupported attribute type for: coordinate_transformation_mode
// 4: Unsupported attribute type for: mode
// 4: Unsupported attribute type for: nearest_mode

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
    default:
        return TensorDataType::UNDEFINED;
    }
}

Tensor parseONNXTensor(const onnx::TensorProto &onnx_tensor)
{
    TensorDataType data_type = convertONNXDataType(onnx_tensor.data_type());

    // Get the dimensions
    std::vector<size_t> dims;
    for (int i = 0; i < onnx_tensor.dims_size(); ++i)
    {
        dims.push_back(onnx_tensor.dims(i));
    }

    // Create a Tensor object
    Tensor tensor(data_type, dims);

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

        // Allocate new data buffer
        void *data = nullptr;
        switch (data_type)
        {
        case TensorDataType::FLOAT32:
            data = new float[num_elements];
            std::memcpy(data, raw_data.data(), expected_size);
            tensor.setDataPointer(static_cast<float *>(data), dims);
            break;
        case TensorDataType::FLOAT64:
            data = new double[num_elements];
            std::memcpy(data, raw_data.data(), expected_size);
            tensor.setDataPointer(static_cast<double *>(data), dims);
            break;
        case TensorDataType::INT32:
            data = new int32_t[num_elements];
            std::memcpy(data, raw_data.data(), expected_size);
            tensor.setDataPointer(static_cast<int32_t *>(data), dims);
            break;
        case TensorDataType::INT64:
            data = new int64_t[num_elements];
            std::memcpy(data, raw_data.data(), expected_size);
            tensor.setDataPointer(static_cast<int64_t *>(data), dims);
            break;
        case TensorDataType::INT8:
            data = new int8_t[num_elements];
            std::memcpy(data, raw_data.data(), expected_size);
            tensor.setDataPointer(static_cast<int8_t *>(data), dims);
            break;
        case TensorDataType::UINT8:
            data = new uint8_t[num_elements];
            std::memcpy(data, raw_data.data(), expected_size);
            tensor.setDataPointer(static_cast<uint8_t *>(data), dims);
            break;
        default:
            throw std::runtime_error("Unsupported data type.");
        }
    }
    else
    {
        // Handle the case where data is stored in repeated fields
        switch (data_type)
        {
        case TensorDataType::FLOAT32:
        {
            size_t num_elements = onnx_tensor.float_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for FLOAT32 data.");
            }
            float *data = new float[num_elements];
            std::copy(onnx_tensor.float_data().begin(), onnx_tensor.float_data().end(), data);
            tensor.setDataPointer(data, dims);
            break;
        }
        case TensorDataType::FLOAT64:
        {
            size_t num_elements = onnx_tensor.double_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for FLOAT64 data.");
            }
            double *data = new double[num_elements];
            std::copy(onnx_tensor.double_data().begin(), onnx_tensor.double_data().end(), data);
            tensor.setDataPointer(data, dims);
            break;
        }
        case TensorDataType::INT32:
        {
            size_t num_elements = onnx_tensor.int32_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for INT32 data.");
            }
            int32_t *data = new int32_t[num_elements];
            std::copy(onnx_tensor.int32_data().begin(), onnx_tensor.int32_data().end(), data);
            tensor.setDataPointer(data, dims);
            break;
        }
        case TensorDataType::INT64:
        {
            size_t num_elements = onnx_tensor.int64_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for INT64 data.");
            }
            int64_t *data = new int64_t[num_elements];
            std::copy(onnx_tensor.int64_data().begin(), onnx_tensor.int64_data().end(), data);
            tensor.setDataPointer(data, dims);
            break;
        }
        case TensorDataType::INT8:
        {
            size_t num_elements = onnx_tensor.int32_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for INT8 data.");
            }
            int8_t *data = new int8_t[num_elements];
            std::transform(onnx_tensor.int32_data().begin(), onnx_tensor.int32_data().end(), data, [](int32_t val)
                           { return static_cast<int8_t>(val); });
            tensor.setDataPointer(data, dims);
            break;
        }
        case TensorDataType::UINT8:
        {
            size_t num_elements = onnx_tensor.int32_data_size();
            if (num_elements != tensor.getNumElements())
            {
                throw std::runtime_error("Mismatch in number of elements for UINT8 data.");
            }
            uint8_t *data = new uint8_t[num_elements];
            std::transform(onnx_tensor.int32_data().begin(), onnx_tensor.int32_data().end(), data, [](int32_t val)
                           { return static_cast<uint8_t>(val); });
            tensor.setDataPointer(data, dims);
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type in type-specific fields.");
        }
    }

    return tensor;
}

Graph parseONNX(const std::string &file_path)
{
    // Create an ONNX model object
    onnx::ModelProto model;

    // Open the file as an input stream
    std::ifstream input(file_path, std::ios::in | std::ios::binary);
    if (!input)
    {
        std::cerr << "Cannot open the file: " << file_path << std::endl;
        throw std::runtime_error("Failed to open file.");
    }

    // Parse the file
    if (!model.ParseFromIstream(&input))
    {
        std::cerr << "Failed to parse ONNX file." << std::endl;
        throw std::runtime_error("Failed to parse ONNX file.");
    }

    // Create a graph object to hold the computation graph
    Graph graph;

    // Extract the graph from the model
    const onnx::GraphProto &onnx_graph = model.graph();

    // Iterate over the nodes in the graph
    for (int i = 0; i < onnx_graph.node_size(); ++i)
    {
        const onnx::NodeProto &onnx_node = onnx_graph.node(i);

        // Create a Node object for each ONNX node
        Node node(onnx_node.name(), onnx_node.op_type());

        // Add inputs to the node
        for (int j = 0; j < onnx_node.input_size(); ++j)
        {
            node.addInput(onnx_node.input(j));
        }

        // Add outputs to the node
        for (int j = 0; j < onnx_node.output_size(); ++j)
        {
            node.addOutput(onnx_node.output(j));
        }

        // Parse attributes if any
        for (int j = 0; j < onnx_node.attribute_size(); ++j)
        {
            const onnx::AttributeProto &attr = onnx_node.attribute(j);

            // Check the type of the attribute using the type() function
            switch (attr.type())
            {
            case onnx::AttributeProto_AttributeType_TENSOR:
            {
                Tensor attribute_tensor = parseONNXTensor(attr.t());
                node.addAttribute(attr.name(), attribute_tensor);
                break;
            }
            case onnx::AttributeProto_AttributeType_INT:
            {
                node.addAttribute(attr.name(), static_cast<int>(attr.i()));
                break;
            }
            case onnx::AttributeProto_AttributeType_FLOAT:
            {
                node.addAttribute(attr.name(), static_cast<float>(attr.f()));
                break;
            }
            case onnx::AttributeProto_AttributeType_INTS:
            {
                std::vector<int64_t> int_vals(attr.ints().begin(), attr.ints().end());
                node.addAttribute(attr.name(), int_vals);
                break;
            }
            case onnx::AttributeProto_AttributeType_FLOATS:
            {
                std::vector<float> float_vals(attr.floats().begin(), attr.floats().end());
                node.addAttribute(attr.name(), float_vals);
                break;
            }
            case onnx::AttributeProto_AttributeType_STRING:
            {
                node.addAttribute(attr.name(), attr.s());
                break;
            }
            default:
                std::cerr << "Unsupported attribute type for: " << attr.name() << std::endl;
                break;
            }
        }

        // Add the node to the graph
        graph.addNode(node);
    }

    return graph;
}
