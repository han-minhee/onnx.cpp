#ifndef ENUMS_HPP
#define ENUMS_HPP

enum class BufferOperationResult
{
    SUCCESS,
    FAILURE
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

    /// XXX: Should "half" included in the CPU backend? AVX512 can use float16, but what about non-AVX512 implementations?
    FLOAT16
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
    MaxPool
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
    DEVICE_UNSUPPORTED
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