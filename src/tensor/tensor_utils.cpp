#include "tensor/tensor.hpp"
#include "enums.hpp"

namespace TensorUtils
{
    size_t getDataTypeSize(TensorDataType dtype)
    {
        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            return sizeof(float);
        case TensorDataType::FLOAT64:
            return sizeof(double);
        case TensorDataType::INT32:
            return sizeof(int32_t);
        case TensorDataType::INT64:
            return sizeof(int64_t);
        case TensorDataType::INT8:
            return sizeof(int8_t);
        case TensorDataType::UINT8:
            return sizeof(uint8_t);

        // custom types
        case TensorDataType::FLOAT16:
            return sizeof(half_t);

        case TensorDataType::UNDEFINED:
            return 0;
        default:
            throw std::runtime_error("Unsupported data type in getDataTypeSize");
        }
    }
    std::string getDataTypeName(TensorDataType dtype)
    {
        switch (dtype)
        {
        case TensorDataType::FLOAT32:
            return "FLOAT32";
        case TensorDataType::FLOAT64:
            return "FLOAT64";
        case TensorDataType::INT32:
            return "INT32";
        case TensorDataType::INT64:
            return "INT64";
        case TensorDataType::INT8:
            return "INT8";
        case TensorDataType::UINT8:
            return "UINT8";

        // custom types
        case TensorDataType::FLOAT16:
            return "FLOAT16";

        case TensorDataType::UNDEFINED:
            return "UNDEFINED";
        default:
            throw std::runtime_error("Unsupported data type in getDataTypeName");
        }
    }

    TensorCompareResult areTensorsEqual(const Tensor &left_tensor, const Tensor &right_tensor)
    {
        // make lhs and rhs from the same dimensions and data type
        TensorDataType lhs_data_type = left_tensor.getDataType();
        TensorDataType rhs_data_type = right_tensor.getDataType();

        // get the dims
        std::vector<size_t> lhs_dims = left_tensor.getDims();
        std::vector<size_t> rhs_dims = right_tensor.getDims();

        Tensor lhs(lhs_data_type, lhs_dims);
        Tensor rhs(rhs_data_type, rhs_dims);

        // copy the data
        lhs.copyFrom(left_tensor);
        rhs.copyFrom(right_tensor);

        // check if the data types are the same
        if (lhs.getDataType() != rhs.getDataType())
        {
            return TensorCompareResult::DATA_TYPE_MISMATCH;
        }

        // check if the dimensions are the same
        if (lhs.getDims() != rhs.getDims())
        {
            return TensorCompareResult::SHAPE_MISMATCH;
        }

        switch (lhs.getDataType())
        {
        case TensorDataType::FLOAT32:
        {
            const float *lhs_data = lhs.data<float>();
            const float *rhs_data = rhs.data<float>();

            // get the largest of the absolute values
            float max_val = 0.0f;
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                float abs_val = std::abs(lhs_data[i]);
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                }
            }
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                float abs_val = std::abs(rhs_data[i]);
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                }
            }

            float tolerance = max_val < 1.0f ? 1e-3f : max_val * 1e-3f;

            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (std::abs(lhs_data[i] - rhs_data[i]) > tolerance)
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        case TensorDataType::FLOAT64:
        {
            const double *lhs_data = lhs.data<double>();
            const double *rhs_data = rhs.data<double>();

            // get the largest of the absolute values
            double max_val = 0.0;
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                double abs_val = std::abs(lhs_data[i]);
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                }
            }
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                double abs_val = std::abs(rhs_data[i]);
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                }
            }

            double tolerance = max_val < 1.0 ? 1e-3 : max_val * 1e-3;

            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (std::abs(lhs_data[i] - rhs_data[i]) > tolerance)
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        case TensorDataType::INT32:
        {
            const int32_t *lhs_data = lhs.data<int32_t>();
            const int32_t *rhs_data = rhs.data<int32_t>();
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (lhs_data[i] != rhs_data[i])
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        case TensorDataType::INT64:
        {
            const int64_t *lhs_data = lhs.data<int64_t>();
            const int64_t *rhs_data = rhs.data<int64_t>();
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (lhs_data[i] != rhs_data[i])
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        case TensorDataType::INT8:
        {
            const int8_t *lhs_data = lhs.data<int8_t>();
            const int8_t *rhs_data = rhs.data<int8_t>();
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (lhs_data[i] != rhs_data[i])
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        case TensorDataType::UINT8:
        {
            const uint8_t *lhs_data = lhs.data<uint8_t>();
            const uint8_t *rhs_data = rhs.data<uint8_t>();
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (lhs_data[i] != rhs_data[i])
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }
        // custom types
        case TensorDataType::FLOAT16:
        {
            const half_t *lhs_data = lhs.data<half_t>();
            const half_t *rhs_data = rhs.data<half_t>();
            for (size_t i = 0; i < lhs.getNumElements(); ++i)
            {
                if (lhs_data[i] != rhs_data[i])
                {
                    return TensorCompareResult::DATA_MISMATCH;
                }
            }
            break;
        }

        default:
            throw std::runtime_error("Unsupported data type in areTensorsEqual");
        }
        return TensorCompareResult::EQUAL;
    }

    std::string TensorCompareResultToString(TensorCompareResult result)
    {
        switch (result)
        {
        case TensorCompareResult::EQUAL:
            return "EQUAL";
        case TensorCompareResult::SHAPE_MISMATCH:
            return "SHAPE_MISMATCH";
        case TensorCompareResult::DATA_TYPE_MISMATCH:
            return "DATA_TYPE_MISMATCH";
        case TensorCompareResult::DATA_MISMATCH:
            return "DATA_MISMATCH";
        default:
            throw std::runtime_error("Unsupported TensorCompareResult in TensorCompareResultToString");
        }
    }
}