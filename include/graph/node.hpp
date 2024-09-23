#ifndef NODE_HPP
#define NODE_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <variant>
#include "tensor/tensor.hpp"

// namespace PrintUtil{
//     // print the attributes
//     template <typename T>
//     void printAttribute(const std::string &key, const T &value)
//     {
//         std::cout << key << ": " << value << std::endl;
//     }
// }

class Node
{
public:
    Node(const std::string &name, const std::string &op_type);

    void addInput(const std::string &input_name);
    void addOutput(const std::string &output_name);

    using AttributeValue = std::variant<int64_t, float, std::string, std::vector<int>, std::vector<float>, std::vector<int64_t>, Tensor>;

    void addAttribute(const std::string &key, const AttributeValue &value);

    const std::string &getName() const;
    const std::string &getOpType() const;
    const std::vector<std::string> &getInputs() const;
    const std::vector<std::string> &getOutputs() const;
    const std::unordered_map<std::string, AttributeValue> &getAttributes() const;

    template <typename T>
    std::optional<T> getAttribute(const std::string &key) const;

    std::string toString() const;

private:
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, AttributeValue> attributes;
};

#endif // NODE_HPP
