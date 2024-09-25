#ifndef NODE_HPP
#define NODE_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <variant>
#include "tensor/tensor.hpp"
#include "operator/operator.hpp"

class Node
{
public:
    using AttributeValue = std::variant<int64_t, float, std::string, std::vector<int>, std::vector<float>, std::vector<int64_t>, Tensor>;

    Node(const std::string &name, const std::string &op_type);
    void addInput(const std::string &input_name);
    void addOutput(const std::string &output_name);
    void addAttribute(const std::string &key, const AttributeValue &value);

    const std::string &getName() const;
    const std::string &getOpType() const;
    const std::vector<std::string> &getInputNames() const;
    const std::vector<std::string> &getOutputNames() const;
    const std::unordered_map<std::string, AttributeValue> &getAttributes() const;

    template <typename T>
    std::optional<T> getAttribute(const std::string &key) const;

    std::string toString() const;

    OperatorExecuteResult execute(const std::vector<Tensor> &inputs, std::vector<Tensor *> &outputs);

private:
    std::string name;
    std::string op_type;

    // inside a session, there will be a map of <string,Operator*> to get the operator
    Operator *op;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::unordered_map<std::string, AttributeValue> attributes;
};

#endif // NODE_HPP
