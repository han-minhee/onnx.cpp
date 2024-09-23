#ifndef NPY_PARSER_HPP
#define NPY_PARSER_HPP

#include <string>
#include "tensor/tensor.hpp"

class NpyParser
{
public:
    static Tensor load(const std::string &file_path);

private:
    static void parseHeader(FILE *file, TensorDataType &dtype, std::vector<size_t> &dims, bool &fortran_order);

    static TensorDataType determineDataType(const std::string &dtype_str);
};

#endif // NPY_PARSER_HPP
