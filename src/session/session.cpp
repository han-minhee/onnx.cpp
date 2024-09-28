#include <stdexcept>
#include <iostream>
#include <fstream>

#include "onnx.pb.h"

#include "session/session.hpp"
#include "operator/operator.hpp"
#include "operator/operators.hpp"
#include "tensor/tensor.hpp"
#include "parser/onnx_parser.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "session/session.hpp"
#include "parser/npy_parser.hpp"
#include "operator/operator_registry.hpp"

#include "tensor/tensor_utils.hpp"
#include "enums.hpp"

Session::Session(const std::string &onnx_file_path, SessionConfig config = SessionConfig())
{
    hostDevice = new CpuDevice();
    sessionConfig = config;

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
        Tensor tensor = parseONNXTensor(onnx_initializer, hostDevice); // CPU device is used for parsing now
        tensor.to(sessionConfig.device);                        // move the tensor to the selected device
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
                Tensor attribute_tensor = parseONNXTensor(attr.t(), hostDevice); // CPU device is used for parsing now
                attribute_tensor.to(sessionConfig.device);               // move the tensor to the selected device
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
    const OperatorRegistry::OperatorFunctions *op = OperatorRegistry::getOperatorFunctions(node.getOpType());
    std::vector<Tensor> input_tensors;
    for (const auto &input_name : node.getInputNames())
    {
        if (input_name.empty())
        {
            input_tensors.push_back(Tensor(sessionConfig.device));
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

    for (size_t i = 0; i < node.getOutputNames().size(); ++i)
    {
        const std::string &output_name = node.getOutputNames()[i];
        const std::vector<size_t> &shape = output_shapes[i];
        TensorDataType dtype = output_data_types[i];

        Tensor &output_tensor = getOrAllocateIntermediateTensor(output_name, shape, dtype);

        output_tensors.push_back(&output_tensor);
    }

    OperatorExecuteResult result = op->execute(input_tensors, output_tensors, node.getAttributes(), sessionConfig.device);
    if (result != OperatorExecuteResult::SUCCESS)
    {
        throw std::runtime_error("Operator execution failed for node: " + node.getName());
    }
    for (size_t i = 0; i < node.getOutputNames().size(); ++i)
    {
        const std::string &output_name = node.getOutputNames()[i];
        tensorMap[output_name] = *output_tensors[i];
    }
}
Tensor &Session::getOrAllocateIntermediateTensor(const std::string &name, const std::vector<size_t> &dims, TensorDataType dtype)
{

    if (intermediateTensorPool.find(name) != intermediateTensorPool.end())
    {
        return intermediateTensorPool[name];
    }
    else
    {

        intermediateTensorPool[name] = Tensor(dtype, dims, sessionConfig.device);

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
    const OperatorRegistry::OperatorFunctions *op = OperatorRegistry::getOperatorFunctions(node.getOpType());
    std::vector<Tensor> input_tensors;
    for (const auto &input_name : node.getInputNames())
    {
        if (input_name.empty())
        {
            input_tensors.push_back(Tensor(sessionConfig.device));
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
    for (size_t i = 0; i < node.getOutputNames().size(); ++i)
    {
        const std::string &output_name = node.getOutputNames()[i];
        const std::vector<size_t> &shape = output_shapes[i];
        TensorDataType dtype = output_data_types[i];
        Tensor &output_tensor = getOrAllocateIntermediateTensor(output_name, shape, dtype);
        output_tensors.push_back(&output_tensor);
    }
    OperatorExecuteResult result = op->execute(input_tensors, output_tensors, node.getAttributes(), sessionConfig.device);
    if (result != OperatorExecuteResult::SUCCESS)
    {
        throw std::runtime_error("Operator execution failed for node: " + node.getName());
    }
    for (size_t i = 0; i < node.getOutputNames().size(); ++i)
    {
        const std::string &output_name = node.getOutputNames()[i];
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
    TensorCompareResult is_equal = TensorUtils::areTensorsEqual(output_tensor, expected_tensor);
    if (is_equal == TensorCompareResult::EQUAL)
    {
        std::cout << "Output tensor " << TensorUtils::TensorCompareResultToString(is_equal) << ":" << sanitized_output_name << std::endl;
        return;
    }
    else
    {
        std::cerr << "Mismatch at Node: " << node.getName() << std::endl;
        std::cerr << "Output tensor " << TensorUtils::TensorCompareResultToString(is_equal) << ":" << sanitized_output_name << std::endl;
        std::cerr << "Expected tensor: " << std::endl;
        std::cerr << expected_tensor.toString() << std::endl;
        std::cerr << "Actual tensor: " << std::endl;
        std::cerr << output_tensor.toString() << std::endl;
    }
}