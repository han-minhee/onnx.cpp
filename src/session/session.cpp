#include <stdexcept>
#include <iostream>
#include <fstream>

#include "onnx/onnx.proto3.pb.h"

#include "session/session.hpp"
#include "operator/operator.hpp"
#include "operator/operators.hpp"
#include "tensor/tensor.hpp"
#include "parser/onnx_parser.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "session/session.hpp"
#include "parser/npy_parser.hpp"

Session::Session(const std::string &onnx_file_path)
{
    onnx::ModelProto model;
    std::ifstream input(onnx_file_path, std::ios::in | std::ios::binary);
    if (!input)
    {
        std::cerr << "Cannot open the file: " << onnx_file_path << std::endl;
        throw std::runtime_error("Failed to open file.");
    }
    if (!model.ParseFromIstream(&input))
    {
        std::cerr << "Failed to parse ONNX file." << std::endl;
        throw std::runtime_error("Failed to parse ONNX file.");
    }
    Graph temp_graph;
    const onnx::GraphProto &onnx_graph = model.graph();
    for (int i = 0; i < onnx_graph.input_size(); ++i)
    {
        const onnx::ValueInfoProto &input = onnx_graph.input(i);
        graphInputNames.push_back(input.name());
    }
    for (int i = 0; i < onnx_graph.output_size(); ++i)
    {
        const onnx::ValueInfoProto &output = onnx_graph.output(i);
        graphOutputNames.push_back(output.name());
    }
    for (int i = 0; i < onnx_graph.initializer_size(); ++i)
    {
        const onnx::TensorProto &onnx_initializer = onnx_graph.initializer(i);
        Tensor tensor = parseONNXTensor(onnx_initializer);
        tensorMap[onnx_initializer.name()] = tensor;
    }
    for (int i = 0; i < onnx_graph.node_size(); ++i)
    {
        const onnx::NodeProto &onnx_node = onnx_graph.node(i);
        Node node(onnx_node.name(), onnx_node.op_type());
        for (int j = 0; j < onnx_node.input_size(); ++j)
        {
            node.addInput(onnx_node.input(j));
        }
        for (int j = 0; j < onnx_node.output_size(); ++j)
        {
            node.addOutput(onnx_node.output(j));
        }
        for (int j = 0; j < onnx_node.attribute_size(); ++j)
        {
            const onnx::AttributeProto &attr = onnx_node.attribute(j);
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
                node.addAttribute(attr.name(), static_cast<int64_t>(attr.i()));
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
        temp_graph.addNode(node);
    }
    temp_graph.topologicalSort();
    this->graph = temp_graph;
}
std::unordered_map<std::string, Tensor> Session::run(const std::unordered_map<std::string, Tensor> &inputs)
{
    prepareExecution(inputs);
    const auto &nodes = graph.getNodes();
    for (const auto &node : nodes)
    {
        // std::cout << "Executing node: " << node.getName() << std::endl;
        executeNode(node);
    }
    std::unordered_map<std::string, Tensor> outputs;
    for (const auto &output_name : graphOutputNames)
    {
        outputs[output_name] = tensorMap.at(output_name);
    }
    return outputs;
}
void Session::prepareExecution(const std::unordered_map<std::string, Tensor> &inputs)
{
    for (const auto &input_name : graphInputNames)
    {
        if (inputs.find(input_name) == inputs.end())
        {
            throw std::runtime_error("Missing required input tensor: " + input_name);
        }
        tensorMap[input_name] = inputs.at(input_name);
    }
}
void Session::executeNode(const Node &node)
{
    // std::cout << "Executing node: " << node.getName() << std::endl;
    std::unique_ptr<Operator> op = OperatorFactory::createOperator(node.getOpType());
    std::vector<Tensor> input_tensors;
    for (const auto &input_name : node.getInputs())
    {
        // std::cout << "Getting input tensor: " << input_name << std::endl;
        if (input_name.empty())
        {
            // std::cout << "Input tensor name is blank" << std::endl;
            input_tensors.push_back(Tensor());
            continue;
        }
        if (tensorMap.find(input_name) == tensorMap.end())
        {
            throw std::runtime_error("Input tensor not found: " + input_name);
        }
        input_tensors.push_back(tensorMap.at(input_name));
    }
    // std::cout << "Found " << input_tensors.size() << " input tensors:" << std::endl;
    // for (const auto &input_tensor : input_tensors)
    // {
    //     std::cout << input_tensor.toString() << std::endl;
    // }
    // std::cout << "Infering output shapes" << std::endl;
    std::vector<std::vector<size_t>> output_shapes = op->inferOutputShapes(input_tensors, node.getAttributes());
    // std::cout << "Inferring output data type" << std::endl;
    std::vector<TensorDataType> output_data_types = op->inferOutputDataTypes(input_tensors, node.getAttributes());
    // std::cout << "Inferred output shapes and data types" << std::endl;
    // std::cout << "Output shapes: " << std::endl;
    // for (const auto &shape : output_shapes)
    // {
    //     for (size_t dim : shape)
    //     {
    //         std::cout << dim << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "Allocating output tensors" << std::endl;
    std::vector<Tensor *> output_tensors;
    // std::cout << "Dtypes: " << std::endl;
    for (TensorDataType dtype : output_data_types)
    {
        std::cout << TensorUtils::getDataTypeName(dtype) << std::endl;
    }
    // std::cout << "Output shapes: " << std::endl;
    for (size_t i = 0; i < node.getOutputs().size(); ++i)
    {
        const std::string &output_name = node.getOutputs()[i];
        const std::vector<size_t> &shape = output_shapes[i];
        TensorDataType dtype = output_data_types[i];
        // std::cout << "Allocating output tensor: " << output_name << std::endl;
        // std::cout << "For dtype: " << TensorUtils::getDataTypeName(dtype) << std::endl;
        Tensor &output_tensor = getOrAllocateIntermediateTensor(output_name, shape, dtype);
        // std::cout << "Allocated output tensor: " << output_name << std::endl;
        output_tensors.push_back(&output_tensor);
    }
    // std::cout << "Executing operator: " << node.getOpType() << std::endl;
    OperatorExecuteResult result = op->execute(input_tensors, output_tensors, node.getAttributes());
    if (result != OperatorExecuteResult::SUCCESS)
    {
        throw std::runtime_error("Operator execution failed for node: " + node.getName());
    }
    for (size_t i = 0; i < node.getOutputs().size(); ++i)
    {
        const std::string &output_name = node.getOutputs()[i];
        tensorMap[output_name] = *output_tensors[i];
        // std::cout << "Output tensor: " << output_name << std::endl;
        // std::cout << output_tensors[i]->toString() << std::endl;
    }
    // std::cout << "Stored output tensors in tensorMap" << std::endl;
}
Tensor &Session::getOrAllocateIntermediateTensor(const std::string &name, const std::vector<size_t> &dims, TensorDataType dtype)
{
    // std::cout << "Getting or allocating intermediate tensor: " << name << std::endl;
    if (intermediateTensorPool.find(name) != intermediateTensorPool.end())
    {
        return intermediateTensorPool[name];
    }
    else
    {
        // std::cout << "Dims: " << std::endl;
        // for (size_t dim : dims)
        // {
        //     std::cout << dim << " ";
        // }
        intermediateTensorPool[name] = Tensor(dtype, dims);
        // std::cout << "Intermediate tensor allocated" << std::endl;
        return intermediateTensorPool[name];
    }
}
Tensor &Session::getTensorByName(const std::string &name)
{
    if (tensorMap.find(name) == tensorMap.end())
    {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return tensorMap[name];
}
std::unordered_map<std::string, Tensor> Session::runWithValidation(const std::unordered_map<std::string, Tensor> &inputs)
{
    prepareExecution(inputs);
    const auto &nodes = graph.getNodes();
    for (const auto &node : nodes)
    {
        std::cout << "\nExecuting node: " << node.getName() << std::endl;
        executeAndValidateNode(node);
    }
    std::unordered_map<std::string, Tensor> outputs;
    for (const auto &output_name : graphOutputNames)
    {
        outputs[output_name] = tensorMap.at(output_name);
    }
    return outputs;
}
std::string sanitizeFileName(const std::string &name)
{
    std::string sanitized = name;
    std::replace(sanitized.begin(), sanitized.end(), '/', '.');
    std::replace(sanitized.begin(), sanitized.end(), ':', '.');
    sanitized.erase(0, sanitized.find_first_not_of('.'));
    return sanitized;
}
void Session::executeAndValidateNode(const Node &node)
{
    std::unique_ptr<Operator> op = OperatorFactory::createOperator(node.getOpType());
    std::vector<Tensor> input_tensors;
    for (const auto &input_name : node.getInputs())
    {
        if (input_name.empty())
        {
            input_tensors.push_back(Tensor());
            continue;
        }
        if (tensorMap.find(input_name) == tensorMap.end())
        {
            throw std::runtime_error("Input tensor not found: " + input_name);
        }
        input_tensors.push_back(tensorMap.at(input_name));
    }
    std::vector<std::vector<size_t>> output_shapes = op->inferOutputShapes(input_tensors, node.getAttributes());
    std::vector<TensorDataType> output_data_types = op->inferOutputDataTypes(input_tensors, node.getAttributes());
    std::vector<Tensor *> output_tensors;
    for (size_t i = 0; i < node.getOutputs().size(); ++i)
    {
        const std::string &output_name = node.getOutputs()[i];
        const std::vector<size_t> &shape = output_shapes[i];
        TensorDataType dtype = output_data_types[i];
        Tensor &output_tensor = getOrAllocateIntermediateTensor(output_name, shape, dtype);
        output_tensors.push_back(&output_tensor);
    }
    OperatorExecuteResult result = op->execute(input_tensors, output_tensors, node.getAttributes());
    if (result != OperatorExecuteResult::SUCCESS)
    {
        throw std::runtime_error("Operator execution failed for node: " + node.getName());
    }
    for (size_t i = 0; i < node.getOutputs().size(); ++i)
    {
        const std::string &output_name = node.getOutputs()[i];
        tensorMap[output_name] = *output_tensors[i];
        std::string sanitized_name = sanitizeFileName(output_name);
        compareOutputToReference(node, sanitized_name, *output_tensors[i]);
    }
}
void Session::compareOutputToReference(const Node &node, const std::string &sanitized_output_name, const Tensor &output_tensor)
{
    std::string reference_path = "../tests/data/npy/" + sanitized_output_name + ".npy";
    std::ifstream infile(reference_path);
    if (!infile.good())
    {
        std::cout << "Reference file not found for output: " << sanitized_output_name << std::endl;
        return;
    }
    Tensor expected_tensor = NpyParser::load(reference_path);
    TensorUtils::TensorCompareResult is_equal = TensorUtils::areTensorsEqual(output_tensor, expected_tensor);
    if (is_equal == TensorUtils::TensorCompareResult::EQUAL)
    {
        std::cout << "Output tensor " << TensorCompareResultToString(is_equal) << ":" << sanitized_output_name << std::endl;
        return;
    }
    else{
        std::cerr << "Mismatch at Node: " << node.getName() << std::endl;
        std::cerr << "Output tensor " << TensorCompareResultToString(is_equal) << ":" << sanitized_output_name << std::endl;
        std::cerr << "Expected tensor: " << std::endl;
        std::cerr << expected_tensor.toString() << std::endl;
        std::cerr << "Actual tensor: " << std::endl;
        std::cerr << output_tensor.toString() << std::endl;
    }
}