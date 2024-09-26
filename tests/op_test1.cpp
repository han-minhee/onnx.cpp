// #include <gtest/gtest.h>
// #include "tensor/tensor.hpp"
// #include "operator/operators.hpp"

// void PrintTo(OperatorExecuteResult result, std::ostream *os)
// {
//     *os << OperatorUtils::OperatorExecuteResultToString(result);
// }
// // Utility function
// void run_and_check_operator(Operator &op,
//                             const std::vector<Tensor> &inputs,
//                             std::vector<Tensor *> outputs,
//                             const std::vector<Tensor> &expected,
//                             std::unordered_map<std::string, Node::AttributeValue> attributes = {},
//                             OperatorExecuteResult expected_execute_result = OperatorExecuteResult::SUCCESS,
//                             DeviceType deviceType = DeviceType::CPU)

// {
//     OperatorExecuteResult result_code = op.execute(inputs, outputs, attributes, deviceType);
//     ASSERT_EQ(result_code, expected_execute_result);

//     if (result_code != OperatorExecuteResult::SUCCESS)
//         return;

//     ASSERT_EQ(outputs.size(), expected.size());
//     for (size_t i = 0; i < outputs.size(); i++)
//     {
//         ASSERT_EQ(outputs[i]->getDims(), expected[i].getDims());
//         ASSERT_EQ(outputs[i]->getDataType(), expected[i].getDataType());

//         switch (outputs[i]->getDataType())
//         {
//         case TensorDataType::FLOAT32:
//         {
//             const float *output_data = outputs[i]->data<float>();
//             const float *expected_data = expected[i].data<float>();

//             for (size_t j = 0; j < expected[i].getNumElements(); j++)
//             {
//                 ASSERT_NEAR(output_data[j], expected_data[j], 1e-4);
//             }
//             break;
//         }
//         case TensorDataType::INT32:
//         {
//             const int32_t *output_data = outputs[i]->data<int32_t>();
//             const int32_t *expected_data = expected[i].data<int32_t>();

//             for (size_t j = 0; j < expected[i].getNumElements(); j++)
//             {
//                 ASSERT_EQ(output_data[j], expected_data[j]);
//             }
//             break;
//         }
//         case TensorDataType::INT64:
//         {
//             const int64_t *output_data = outputs[i]->data<int64_t>();
//             const int64_t *expected_data = expected[i].data<int64_t>();

//             for (size_t j = 0; j < expected[i].getNumElements(); j++)
//             {
//                 ASSERT_EQ(output_data[j], expected_data[j]);
//             }
//             break;
//         }
//         default:
//             throw std::runtime_error("Unsupported data type.");
//         }
//     }
// }

// // -------------------- AddOperator Tests --------------------
// TEST(OperatorTest1, AddOperatorBasic)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {6.0f, 8.0f, 10.0f, 12.0f});
//     Tensor output;

//     AddOperator add_op;
//     run_and_check_operator(add_op, {t1, t2}, {&output}, {expected});
// }

// // Data type mismatch
// TEST(OperatorTest1, AddOperatorDataTypeError)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor t2 = create_tensor(TensorDataType::INT32, {2, 2}, {5, 6, 7, 8});
//     Tensor output;

//     AddOperator add_op;
//     run_and_check_operator(add_op, {t1, t2}, {&output}, {}, {}, OperatorExecuteResult::DATA_TYPE_ERROR);
// }

// // Shape mismatch
// TEST(OperatorTest1, AddOperatorShapeMismatchError)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {3, 2}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
//     Tensor output;

//     AddOperator add_op;
//     run_and_check_operator(add_op, {t1, t2}, {&output}, {}, {}, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
// }

// // -------------------- MulOperator Tests --------------------
// TEST(OperatorTest1, MulOperatorBasic)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 12.0f, 21.0f, 32.0f});
//     Tensor output;

//     MulOperator mul_op;
//     run_and_check_operator(mul_op, {t1, t2}, {&output}, {expected});
// }

// // Data type mismatch
// TEST(OperatorTest1, MulOperatorDataTypeError)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor t2 = create_tensor(TensorDataType::INT32, {2, 2}, {5, 6, 7, 8});
//     Tensor output;

//     MulOperator mul_op;
//     run_and_check_operator(mul_op, {t1, t2}, {&output}, {}, {}, OperatorExecuteResult::DATA_TYPE_ERROR);
// }

// // -------------------- SubOperator Tests --------------------
// TEST(OperatorTest1, SubOperatorBasic)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 7.0f, 9.0f, 11.0f});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {4.0f, 5.0f, 6.0f, 7.0f});
//     Tensor output;

//     SubOperator sub_op;
//     run_and_check_operator(sub_op, {t1, t2}, {&output}, {expected});
// }

// // -------------------- DivOperator Tests --------------------
// // TEST(OperatorTest1, DivOperatorDivideByZero)
// // {
// //     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
// //     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 0.0f, 1.0f, 0.0f});

// //     DivOperator div_op;
// //     Tensor output;

// //     // Expecting the operator to handle divide by zero errors
// //     run_and_check_operator(div_op, {t1, t2}, {&output}, {}, {}, OperatorExecuteResult::INPUT_TENSOR_VALUE_ERROR);
// // }

// // -------------------- MatMulOperator Tests --------------------
// TEST(OperatorTest1, MatMulOperatorShapeMismatchError)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1, 2, 3, 4, 5, 6});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {7, 8, 9, 10, 11, 12});
//     Tensor output;

//     MatMulOperator matmul_op;
//     run_and_check_operator(matmul_op, {t1, t2}, {&output}, {}, {}, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
// }

// // -------------------- ConvOperator Tests --------------------
// TEST(OperatorTest1, ConvOperatorInvalidKernelShape)
// {
//     Tensor X = create_tensor(TensorDataType::FLOAT32, {1, 1, 4, 4}, std::vector<float>(16, 1.0f));
//     Tensor W = create_tensor(TensorDataType::FLOAT32, {1, 1, 5, 5}, std::vector<float>(25, 1.0f));
//     Tensor B = create_tensor(TensorDataType::FLOAT32, {1}, {0});
//     Tensor output;

//     ConvOperator conv_op;
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["dilations"] = std::vector<int64_t>{1, 1};
//     attributes["group"] = 1;
//     attributes["kernel_shape"] = std::vector<int64_t>{5, 5}; // Invalid for input size 4x4
//     attributes["pads"] = std::vector<int64_t>{0, 0, 0, 0};
//     attributes["strides"] = std::vector<int64_t>{1, 1};

//     run_and_check_operator(conv_op, {X, W, B}, {&output}, {}, attributes, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
// }

// // -------------------- SigmoidOperator Tests --------------------
// TEST(OperatorTest1, SigmoidOperatorBasic)
// {
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 2}, {0.0f, -1.0f, 1.0f, 2.0f});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2},
//                                     {0.5f, 0.26894f, 0.73106f, 0.880797f});
//     Tensor output;

//     SigmoidOperator sigmoid_op;
//     run_and_check_operator(sigmoid_op, {data}, {&output}, {expected});
// }

// // -------------------- ConstantOperator Tests --------------------
// TEST(OperatorTest1, ConstantOperatorAttributeError)
// {
//     ConstantOperator const_op;
//     Tensor output;

//     // No value attribute provided
//     run_and_check_operator(const_op, {}, {&output}, {}, {}, OperatorExecuteResult::ATTRIBUTE_ERROR);
// }

// // -------------------- SplitOperator Tests --------------------
// TEST(OperatorTest1, SplitOperatorShapeMismatchError)
// {
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
//     Tensor split = create_tensor(TensorDataType::INT64, {2}, {3, 2});
//     Tensor output1, output2;

//     SplitOperator split_op;
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["axis"] = 1;
//     run_and_check_operator(split_op, {data, split}, {&output1, &output2}, {}, attributes, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
// }

// // -------------------- ConcatOperator Tests --------------------
// TEST(OperatorTest1, ConcatOperatorShapeMismatchError)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1, 2, 3, 4});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {5, 6, 7, 8, 9, 10});
//     Tensor output;

//     ConcatOperator concat_op;
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["axis"] = 0;

//     run_and_check_operator(concat_op, {t1, t2}, {&output}, {}, attributes, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
// }