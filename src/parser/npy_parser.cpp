#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <regex>
#include "parser/npy_parser.hpp"

// Function to load .npy file into a Tensor
Tensor NpyParser::load(const std::string &file_path)
{
    // Open the .npy file in binary mode
    FILE *file = fopen(file_path.c_str(), "rb");
    if (!file)
    {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    // Read and parse the header
    TensorDataType dtype;
    std::vector<size_t> dims;
    bool fortran_order;
    parseHeader(file, dtype, dims, fortran_order);

    // Create a Tensor based on parsed dtype and dimensions
    Tensor tensor(dtype, dims);

    // Read the data from the file
    size_t num_elements = tensor.getNumElements();
    if (fortran_order)
    {
        throw std::runtime_error("Fortran order arrays are not supported.");
    }

    // Load data depending on dtype
    switch (dtype)
    {
    case TensorDataType::FLOAT32:
    {
        fread(tensor.data<float>(), sizeof(float), num_elements, file);
        break;
    }
    case TensorDataType::FLOAT64:
    {
        fread(tensor.data<double>(), sizeof(double), num_elements, file);
        break;
    }
    case TensorDataType::INT32:
    {
        fread(tensor.data<int32_t>(), sizeof(int32_t), num_elements, file);
        break;
    }
    case TensorDataType::INT64:
    {
        fread(tensor.data<int64_t>(), sizeof(int64_t), num_elements, file);
        break;
    }
    case TensorDataType::INT8:
    {
        fread(tensor.data<int8_t>(), sizeof(int8_t), num_elements, file);
        break;
    }
    case TensorDataType::UINT8:
    {
        fread(tensor.data<uint8_t>(), sizeof(uint8_t), num_elements, file);
        break;
    }
    // int4_t is currently not implemented for parsing

    // custom types
    case TensorDataType::FLOAT16:
    {
        fread(tensor.data<half_t>(), sizeof(half_t), num_elements, file);
        break;
    }

    // case TensorDataType::BOOL: {
    //     fread(tensor.data<bool>(), sizeof(bool), num_elements, file);
    //     break;
    // }
    default:
        throw std::runtime_error("Unsupported data type for NPY parsing.");
    }

    fclose(file);
    return tensor;
}

// Function to parse the header and extract information about dtype and dimensions
void NpyParser::parseHeader(FILE *file, TensorDataType &dtype, std::vector<size_t> &dims, bool &fortran_order)
{
    // Read the magic string and version number
    char buffer[11];
    fread(buffer, sizeof(char), 6, file); // Magic string: \x93NUMPY
    buffer[6] = '\0';
    if (std::string(buffer, 6) != "\x93NUMPY")
    {
        throw std::runtime_error("Invalid npy file.");
    }

    // Read version numbers
    uint8_t major_version, minor_version;
    fread(&major_version, sizeof(uint8_t), 1, file);
    fread(&minor_version, sizeof(uint8_t), 1, file);

    // Read header length
    uint16_t header_len;
    if (major_version == 1)
    {
        fread(&header_len, sizeof(uint16_t), 1, file);
    }
    else if (major_version == 2)
    {
        uint32_t header_len_32;
        fread(&header_len_32, sizeof(uint32_t), 1, file);
        header_len = header_len_32;
    }
    else
    {
        throw std::runtime_error("Unsupported npy version.");
    }

    // Read the header itself
    std::vector<char> header_buffer(header_len);
    fread(header_buffer.data(), sizeof(char), header_len, file);
    std::string header(header_buffer.begin(), header_buffer.end());

    // Parse the header for dtype, shape, and fortran order using regular expressions
    std::smatch match;
    std::regex dtype_regex("'descr': '([<>=|]?[a-zA-Z0-9]+)'");
    std::regex shape_regex("'shape': \\(([^\\)]*)\\)");
    std::regex fortran_order_regex("'fortran_order': (True|False)");

    // Extract dtype
    if (std::regex_search(header, match, dtype_regex))
    {
        dtype = determineDataType(match[1].str());
    }
    else
    {
        throw std::runtime_error("Failed to parse dtype.");
    }

    // Extract shape
    if (std::regex_search(header, match, shape_regex))
    {
        std::string shape_str = match[1].str();
        std::stringstream ss(shape_str);
        size_t dim;
        while (ss >> dim)
        {
            dims.push_back(dim);
            if (ss.peek() == ',')
                ss.ignore();
        }
    }
    else
    {
        throw std::runtime_error("Failed to parse shape.");
    }

    // Extract fortran_order
    if (std::regex_search(header, match, fortran_order_regex))
    {
        fortran_order = (match[1].str() == "True");
    }
    else
    {
        throw std::runtime_error("Failed to parse fortran_order.");
    }
}

// Function to determine the TensorDataType from the numpy dtype string
TensorDataType NpyParser::determineDataType(const std::string &dtype_str)
{
    if (dtype_str == "<f4" || dtype_str == "|f4")
        return TensorDataType::FLOAT32;
    if (dtype_str == "<f8" || dtype_str == "|f8")
        return TensorDataType::FLOAT64;
    if (dtype_str == "<i4" || dtype_str == "|i4")
        return TensorDataType::INT32;
    if (dtype_str == "<i8" || dtype_str == "|i8")
        return TensorDataType::INT64;
    if (dtype_str == "|i1")
        return TensorDataType::INT8;
    if (dtype_str == "|u1")
        return TensorDataType::UINT8;

    if (dtype_str == "<f2" || dtype_str == "|f2")
        return TensorDataType::FLOAT16;

    // if (dtype_str == "|b1") return TensorDataType::BOOL;
    throw std::runtime_error("Unsupported dtype: " + dtype_str);
}
