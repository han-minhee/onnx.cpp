#ifndef ENUMS_HPP
#define ENUMS_HPP

enum class BufferOperationResult
{
    SUCCESS,
    FAILURE,
    DEVICE_UNSUPPORTED,
    DEVICE_MISMATCH,
    NOT_IMPLEMENTED
};

enum class TensorDataType
{
    UNDEFINED,
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    INT8,
    UINT8,
    FLOAT16
};

enum class TensorCompareResult
{
    EQUAL,
    SHAPE_MISMATCH,
    DATA_TYPE_MISMATCH,
    DATA_MISMATCH
};

enum class OperatorType
{
    Add,
    Mul,
    Sub,
    Div,
    MatMul,
    Conv,
    Sigmoid,
    Constant,
    Split,
    Concat,
    Slice,
    Gather,
    Shape,
    Reshape,
    Softmax,
    Transpose,
    Resize,
    MaxPool,

    MatMulNBits,

    Unknown
};

enum class OperatorExecuteResult
{
    SUCCESS,
    INPUT_TENSOR_ERROR,
    INPUT_TENSOR_VALUE_ERROR,
    OUTPUT_TENSOR_ERROR,
    ATTRIBUTE_ERROR,
    DATA_TYPE_ERROR,
    SHAPE_MISMATCH_ERROR,
    UNSUPPORTED_OPERATION,
    MEMORY_ALLOCATION_ERROR,
    UNKNOWN_ERROR,
    DEVICE_UNSUPPORTED,
    NOT_IMPLEMENTED
    #ifdef USE_HIP
    ,HIP_ERROR
    #endif
};

enum class DeviceType
{
    CPU,
#ifdef USE_HIP
    HIP,
#endif

#ifdef USE_SYCL
    SYCL
#endif

};

#endif // ENUMS_HPP