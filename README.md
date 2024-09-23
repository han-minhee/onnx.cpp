# ONNX.cpp: A simple ONNX engine in Pure C++

## Overview
**The primary goal of the repo is to demonstrate how an ONNX model works and to provide a building block for a custom ONNX engine. This library is currently not recommended for production.** For now, maybe.

There are some DNN implementations built with specific models in mind. While they can be highly optimized for a certain DNN model or hardware, they often lack scalability. General solutions like TensorRT, ONNX Runtime, or TVM generally work well, but as "complete" frameworks, they can be hard to understand or fine-grainly controlled. Projects like llama.cpp or ggml take a more direct approach, but each model needs individual implementation.

This library aims to be simpler while still leveraging the ONNX format, without requiring model-specific implementations. **Currently, this library supports only operators for YOLO, but it's designed to be easily scalable by implementing the required operator classes.**

Minimal dependency was also a consideration. The library uses protobuf to read the ONNX file, but it is used only for parsing the ONNX file and not in other logic. Aside from that, there is no external library dependency other than the standard libraries.

## Implementation
1. The library reads an ONNX file into a `Graph` containing `Node`s (with `NodeAttribute`s and corresponding `Operator`), with weights/bias/constants `Tensor`s initialized inside the `Session`.
2. The `Graph` is topologically sorted, and the execution order of `Node`s is determined.
3. Once the input is provided, the `Session` is run.
4. The `Session` maintains the necessary `Tensor`s and traverses through the `Node`s.
5. When visiting a `Node`, it calls the corresponding `Operator` with input/output tensors and related attribute values.
6. After all `Node`s have been traversed, the `Session` will have all the necessary output `Tensor`s set.
7. The returned `Tensor` with the model output name is the final output.

## Requirements
Other than the basic build environment, protobuf library should be present on the System. I've included the generated pb .cc and .h files.

For Ubuntu, `sudo apt install libprotobuf-dev cmake build-essential ninja-build`

## Usage

### Library Build and Tests
```bash
mkdir build && cd build
cmake -G Ninja ..
cmake --build .
ctest
```

## Issues
- It's partiall intentional, but the performance is horrible.
- Though operators inherit from the same class, their internal implementations are not coherent.
- Just the basic features are enabled for operators, and it doesn't work for some data types or attributes.
- And as usual, needs lots of refactoring.

## Future Work
**There is no optimization at all**. I intend to leave it this way for now so that we can easily understand what each operator does.

### Currently Being Worked on
- [ ] YOLOv8 example
- [ ] GPU Version for AMD/NVIDIA GPUs using HIP

### Future Future Work
- [ ] Support Phi-3 LLM
- [ ] Optimizations and off-loading
- [ ] SYCL implementation for vendor agnostic accelerating
- [ ] General ONNX engine for various models
