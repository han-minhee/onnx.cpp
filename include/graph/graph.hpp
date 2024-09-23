#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "graph/node.hpp"
#include "tensor/tensor.hpp"
#include <vector>
#include <unordered_map>
#include <string>

class Graph
{
public:
    void addNode(const Node &node);
    const std::vector<Node> &getNodes() const;

    void printGraph() const;
    std::string toString() const;

    void topologicalSort();

private:
    std::vector<Node> nodes;
};

#endif // GRAPH_HPP