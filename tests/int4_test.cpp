#include <gtest/gtest.h>

#include "parser/npy_parser.hpp"

#include "tensor/tensor.hpp"
#include "operator/operators.hpp"
#include "operator/operator_registry.hpp"

#include "device/device.hpp"
#include "device/device_cpu.hpp"

#ifdef USE_HIP
#include "device/device_hip.hpp"
#endif

void PrintTo(OperatorExecuteResult result, std::ostream *os)
{
    *os << OperatorUtils::OperatorExecuteResultToString(result);
}

void run_and_check_operator(const OperatorRegistry::OperatorFunctions *op,
                            const std::vector<Tensor> &inputs,
                            std::vector<Tensor *> outputs,
                            std::vector<Tensor> &expected,
                            std::unordered_map<std::string, Node::AttributeValue> attributes,
                            OperatorExecuteResult expected_execute_result,
                            Device *device)
{
    OperatorExecuteResult result_code = op->execute(inputs, outputs, attributes, device);

    ASSERT_EQ(result_code, expected_execute_result);

    if (result_code != OperatorExecuteResult::SUCCESS)
    {
        return;
    }

    ASSERT_EQ(outputs.size(), expected.size());
    for (size_t i = 0; i < outputs.size(); i++)
    {
        ASSERT_EQ(outputs[i]->getDims(), expected[i].getDims());
        ASSERT_EQ(outputs[i]->getDataType(), expected[i].getDataType());

        // move tensors to host
        outputs[i]->to(new CpuDevice());
        expected[i].to(new CpuDevice());

        switch (outputs[i]->getDataType())
        {
        case TensorDataType::FLOAT32:
        {
            const float *output_data = outputs[i]->data<float>();
            const float *expected_data = expected[i].data<float>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_NEAR(output_data[j], expected_data[j], 1e-3);
                // std::cout << output_data[j] << " " << expected_data[j] << std::endl;
            }
            break;
        }
        case TensorDataType::INT32:
        {
            const int32_t *output_data = outputs[i]->data<int32_t>();
            const int32_t *expected_data = expected[i].data<int32_t>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_EQ(output_data[j], expected_data[j]);
            }
            break;
        }
        case TensorDataType::INT64:
        {
            const int64_t *output_data = outputs[i]->data<int64_t>();
            const int64_t *expected_data = expected[i].data<int64_t>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_EQ(output_data[j], expected_data[j]);
            }
            break;
        }

        case TensorDataType::FLOAT16:
        {
            const half_t *output_data = outputs[i]->data<half_t>();
            const half_t *expected_data = expected[i].data<half_t>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_NEAR(output_data[j], expected_data[j], 1e-3);
            }
            break;
        }

        default:
            throw std::runtime_error("Unsupported data type.");
        }
    }
}

#define RUN_TEST_CASE(operator_type, input_tensors, expected_tensors, attributes, expectedResult, device)      \
    do                                                                                                         \
    {                                                                                                          \
        const OperatorRegistry::OperatorFunctions *op = OperatorRegistry::getOperatorFunctions(operator_type); \
        std::vector<std::vector<size_t>> output_shapes = op->inferOutputShapes(input_tensors, attributes);     \
        std::vector<TensorDataType> output_data_types = op->inferOutputDataTypes(input_tensors, attributes);   \
                                                                                                               \
        std::vector<Tensor *> outputs;                                                                         \
        for (size_t i = 0; i < output_shapes.size(); i++)                                                      \
        {                                                                                                      \
            outputs.push_back(new Tensor(output_data_types[i], output_shapes[i], device));                     \
        }                                                                                                      \
                                                                                                               \
        run_and_check_operator(op, input_tensors, outputs, expected_tensors, attributes,                       \
                               expectedResult, device);                                                        \
                                                                                                               \
        for (auto &output : outputs)                                                                           \
        {                                                                                                      \
            delete output;                                                                                     \
        }                                                                                                      \
    } while (0)

// ----------------------- MatMulNBitsOperator Tests -----------------------

// TEST(QuantizationTestCPU, MatMulNBitsBasic0)
// {
//     CpuDevice device = CpuDevice();
//     std::vector<float> A_vector = {
//         0, 1, 2, 3, 4, 5, 6, 7,
//         8, 9, 10, 11, 12, 13, 14, 15,
//         16, 17, 18, 19, 20, 21, 22, 23,
//         24, 25, 26, 27, 28, 29, 30, 31,
//         32, 33, 34, 35, 36, 37, 38, 39,
//         40, 41, 42, 43, 44, 45, 46, 47,
//         48, 49, 50, 51, 52, 53, 54, 55,
//         56, 57, 58, 59, 60, 61, 62, 63,
//         64, 65, 66, 67, 68, 69, 70, 71,
//         72, 73, 74, 75, 76, 77, 78, 79,
//         80, 81, 82, 83, 84, 85, 86, 87,
//         88, 89, 90, 91, 92, 93, 94, 95,
//         96, 97, 98, 99, 100, 101, 102, 103,
//         104, 105, 106, 107, 108, 109, 110, 111,
//         112, 113, 114, 115, 116, 117, 118, 119,
//         120, 121, 122, 123, 124, 125, 126, 127};

//     std::vector<uint8_t> B_vector = {
//         1, 2, 3, 4, 5, 6, 7, 8,
//         9, 10, 11, 12, 13, 14, 15, 16,
//         17, 18, 19, 20, 21, 22, 23, 24,
//         25, 26, 27, 28, 29, 30, 31, 32,
//         33, 34, 35, 36, 37, 38, 39, 40,
//         41, 42, 43, 44, 45, 46, 47, 48,
//         49, 50, 51, 52, 53, 54, 55, 56,
//         57, 58, 59, 60, 61, 62, 63, 64};

//     std::vector<float> S_vector = {0, 1, 2, 3, 4, 5, 6, 7};

//     std::vector<float> expected_vector = {
//         0, -385, -1120, -963, -1984, -1285, -2592, -1351, 0, -1073, -3808, -2643, -6848, -3445, -9120, -3479, 0,
//         -1761, -6496, -4323, -11712, -5605, -15648, -5607, 0, -2449, -9184, -6003, -16576, -7765, -22176, -7735,
//         0, -3137, -11872, -7683, -21440, -9925, -28704, -9863, 0, -3825, -14560, -9363, -26304, -12085, -35232,
//         -11991, 0, -4513, -17248, -11043, -31168, -14245, -41760, -14119, 0, -5201, -19936, -12723, -36032,
//         -16405, -48288, -16247};

//     Tensor A(TensorDataType::FLOAT32, {1, 8, 16}, A_vector, &device);
//     Tensor B(TensorDataType::UINT8, {8, 1, 8}, B_vector, &device);
//     Tensor S = Tensor(TensorDataType::FLOAT32, {8}, S_vector, &device);
//     Tensor expectedOutput(TensorDataType::FLOAT32, {1, 8, 8}, expected_vector, &device);

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     attributes["K"] = static_cast<int64_t>(16);
//     attributes["N"] = static_cast<int64_t>(8);
//     attributes["bits"] = static_cast<int64_t>(4);
//     attributes["block_size"] = static_cast<int64_t>(16);

//     std::vector<Tensor> inputs = {A, B, S};
//     std::vector<Tensor> expected_tensors = {expectedOutput};

//     RUN_TEST_CASE(OperatorType::MatMulNBits, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &device);
// }

// TEST(QuantizationTestCPU, MatMulNBits0)
// {
//     CpuDevice device = CpuDevice();
//     // 2, 128, 3072 for X
//     Tensor X = NpyParser::load("../tests/quantization/model.layers.0.input_layernorm.output_0.npy");
//     // 9216 96, 16 for W
//     Tensor W = NpyParser::load("../tests/quantization/model.layers.0.attn.qkv_proj.MatMul.weight_Q4.npy");
//     // 884736 = 9216 * 96 for S
//     Tensor S = NpyParser::load("../tests/quantization/model.layers.0.attn.qkv_proj.MatMul.weight_scales.npy");
//     Tensor expectedOutput = NpyParser::load("../tests/quantization/model.layers.0.attn.qkv_proj.MatMul.output_0.npy");

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     attributes["K"] = static_cast<int64_t>(3072);
//     attributes["N"] = static_cast<int64_t>(9216);
//     attributes["bits"] = static_cast<int64_t>(4);
//     attributes["block_size"] = static_cast<int64_t>(32);

//     std::vector<Tensor> inputs = {X, W, S};
//     std::vector<Tensor> expected_tensors = {expectedOutput};

//     RUN_TEST_CASE(OperatorType::MatMulNBits, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &device);
// }

// TEST(QuantizationTestCPU, MatMulNBits1)
// {
//     CpuDevice device = CpuDevice();
//     Tensor X = NpyParser::load("../tests/quantization/model.layers.31.post_attention_layernorm.output_0.npy");
//     Tensor W = NpyParser::load("../tests/quantization/model.layers.31.mlp.gate_proj.MatMul.weight_Q4.npy");
//     Tensor S = NpyParser::load("../tests/quantization/model.layers.31.mlp.gate_proj.MatMul.weight_scales.npy");
//     Tensor expectedOutput = NpyParser::load("../tests/quantization/model.layers.31.mlp.gate_proj.MatMul.output_0.npy");

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     attributes["K"] = static_cast<int64_t>(3072);
//     attributes["N"] = static_cast<int64_t>(8192);
//     attributes["bits"] = static_cast<int64_t>(4);
//     attributes["block_size"] = static_cast<int64_t>(32);

//     std::vector<Tensor> inputs = {X, W, S};
//     std::vector<Tensor> expected_tensors = {expectedOutput};

//     RUN_TEST_CASE(OperatorType::MatMulNBits, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &device);
// }

// TEST(QuantizationTestCPU, MatMulNBits2)
// {
//     CpuDevice device = CpuDevice();
//     Tensor X = NpyParser::load("../tests/quantization/model.layers.31.mlp.Mul.output_0.npy");
//     Tensor W = NpyParser::load("../tests/quantization/model.layers.31.mlp.down_proj.MatMul.weight_Q4.npy");
//     Tensor S = NpyParser::load("../tests/quantization/model.layers.31.mlp.down_proj.MatMul.weight_scales.npy");
//     Tensor expectedOutput = NpyParser::load("../tests/quantization/model.layers.31.mlp.down_proj.MatMul.output_0.npy");

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     attributes["K"] = static_cast<int64_t>(8192);
//     attributes["N"] = static_cast<int64_t>(3072);
//     attributes["bits"] = static_cast<int64_t>(4);
//     attributes["block_size"] = static_cast<int64_t>(32);

//     std::vector<Tensor> inputs = {X, W, S};
//     std::vector<Tensor> expected_tensors = {expectedOutput};

//     RUN_TEST_CASE(OperatorType::MatMulNBits, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &device);
// }

#ifdef USE_HIP

TEST(QuantizationTestHIP, MatMulNBitsBasic0)
{
    HipDevice device = HipDevice();
    std::vector<float> A_vector = {
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87,
        88, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103,
        104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 126, 127};

    std::vector<uint8_t> B_vector = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64};

    std::vector<float> S_vector = {0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<float> expected_vector = {
        0, -385, -1120, -963, -1984, -1285, -2592, -1351, 0, -1073, -3808, -2643, -6848, -3445, -9120, -3479, 0,
        -1761, -6496, -4323, -11712, -5605, -15648, -5607, 0, -2449, -9184, -6003, -16576, -7765, -22176, -7735,
        0, -3137, -11872, -7683, -21440, -9925, -28704, -9863, 0, -3825, -14560, -9363, -26304, -12085, -35232,
        -11991, 0, -4513, -17248, -11043, -31168, -14245, -41760, -14119, 0, -5201, -19936, -12723, -36032,
        -16405, -48288, -16247};

    Tensor A(TensorDataType::FLOAT32, {1, 8, 16}, A_vector, &device);
    Tensor B(TensorDataType::UINT8, {8, 1, 8}, B_vector, &device);
    Tensor S = Tensor(TensorDataType::FLOAT32, {8}, S_vector, &device);
    Tensor expectedOutput(TensorDataType::FLOAT32, {1, 8, 8}, expected_vector, &device);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    attributes["K"] = static_cast<int64_t>(16);
    attributes["N"] = static_cast<int64_t>(8);
    attributes["bits"] = static_cast<int64_t>(4);
    attributes["block_size"] = static_cast<int64_t>(16);

    std::vector<Tensor> inputs = {A, B, S};
    std::vector<Tensor> expected_tensors = {expectedOutput};

    RUN_TEST_CASE(OperatorType::MatMulNBits, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &device);
}

TEST(QuantizationTestHIP, MatMulNBits0)
{
    HipDevice device = HipDevice(); // 2, 128, 3072 for X
    Tensor X = NpyParser::load("../tests/quantization/model.layers.0.input_layernorm.output_0.npy");
    // 9216 96, 16 for W
    Tensor W = NpyParser::load("../tests/quantization/model.layers.0.attn.qkv_proj.MatMul.weight_Q4.npy");
    // 884736 = 9216 * 96 for S
    Tensor S = NpyParser::load("../tests/quantization/model.layers.0.attn.qkv_proj.MatMul.weight_scales.npy");
    Tensor expectedOutput = NpyParser::load("../tests/quantization/model.layers.0.attn.qkv_proj.MatMul.output_0.npy");

    // move tensors to device
    X.to(&device);
    W.to(&device);
    S.to(&device);
    expectedOutput.to(&device);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    attributes["K"] = static_cast<int64_t>(3072);
    attributes["N"] = static_cast<int64_t>(9216);
    attributes["bits"] = static_cast<int64_t>(4);
    attributes["block_size"] = static_cast<int64_t>(32);

    std::vector<Tensor> inputs = {X, W, S};
    std::vector<Tensor> expected_tensors = {expectedOutput};

    RUN_TEST_CASE(OperatorType::MatMulNBits, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &device);
}

TEST(QuantizationTestHIP, MatMulNBits1)
{
    HipDevice device = HipDevice();
    Tensor X = NpyParser::load("../tests/quantization/model.layers.31.post_attention_layernorm.output_0.npy");
    Tensor W = NpyParser::load("../tests/quantization/model.layers.31.mlp.gate_proj.MatMul.weight_Q4.npy");
    Tensor S = NpyParser::load("../tests/quantization/model.layers.31.mlp.gate_proj.MatMul.weight_scales.npy");
    Tensor expectedOutput = NpyParser::load("../tests/quantization/model.layers.31.mlp.gate_proj.MatMul.output_0.npy");

    // move tensors to device
    X.to(&device);
    W.to(&device);
    S.to(&device);
    expectedOutput.to(&device);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    attributes["K"] = static_cast<int64_t>(3072);
    attributes["N"] = static_cast<int64_t>(8192);
    attributes["bits"] = static_cast<int64_t>(4);
    attributes["block_size"] = static_cast<int64_t>(32);

    std::vector<Tensor> inputs = {X, W, S};
    std::vector<Tensor> expected_tensors = {expectedOutput};

    RUN_TEST_CASE(OperatorType::MatMulNBits, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &device);
}

TEST(QuantizationTestHIP, MatMulNBits2)
{
    HipDevice device = HipDevice();
    Tensor X = NpyParser::load("../tests/quantization/model.layers.31.mlp.Mul.output_0.npy");
    Tensor W = NpyParser::load("../tests/quantization/model.layers.31.mlp.down_proj.MatMul.weight_Q4.npy");
    Tensor S = NpyParser::load("../tests/quantization/model.layers.31.mlp.down_proj.MatMul.weight_scales.npy");
    Tensor expectedOutput = NpyParser::load("../tests/quantization/model.layers.31.mlp.down_proj.MatMul.output_0.npy");

    // move tensors to device
    X.to(&device);
    W.to(&device);
    S.to(&device);
    expectedOutput.to(&device);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    attributes["K"] = static_cast<int64_t>(8192);
    attributes["N"] = static_cast<int64_t>(3072);
    attributes["bits"] = static_cast<int64_t>(4);
    attributes["block_size"] = static_cast<int64_t>(32);

    std::vector<Tensor> inputs = {X, W, S};
    std::vector<Tensor> expected_tensors = {expectedOutput};

    RUN_TEST_CASE(OperatorType::MatMulNBits, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &device);
}
#endif // USE_HIP