#ifndef SESSION_HPP
#define SESSION_HPP

#include "graph/graph.hpp"
#include "tensor/tensor.hpp"
#include <unordered_map>
#include <string>
#include <memory>

/// XXX: It currently doesn't do anything, but it will be useful in the future
struct SessionConfig
{
    bool enable_optimizations = false;
};

class Session
{
public:
    Session(const std::string &onnx_file_path);
    std::unordered_map<std::string, Tensor> run(const std::unordered_map<std::string, Tensor> &inputs);
    Tensor &getTensorByName(const std::string &name);
    std::unordered_map<std::string, Tensor> runWithValidation(const std::unordered_map<std::string, Tensor> &inputs);
    void executeAndValidateNode(const Node &node);
    void compareOutputToReference(const Node &node, const std::string &sanitized_output_name, const Tensor &output_tensor);
    Tensor &getOrAllocateIntermediateTensor(const std::string &name, const std::vector<size_t> &dims, TensorDataType dtype);

private:
    Graph graph;
    void prepareExecution(const std::unordered_map<std::string, Tensor> &inputs);
    void executeNode(const Node &node);
    std::unordered_map<std::string, Tensor> tensorMap;
    std::unordered_map<std::string, Tensor> intermediateTensorPool;
    std::vector<Node> executionOrder;
    std::vector<std::string> graphInputNames;
    std::vector<std::string> graphOutputNames;
};

std::string sanitizeFileName(const std::string &name);

#endif // SESSION_HPP
