/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <math.h>
// added just for debugging
#include <iostream>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/simple_lstm/models/simple_lstm_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
using SimpleLSTMOpResolver = tflite::MicroMutableOpResolver<7>;

TfLiteStatus RegisterOps(SimpleLSTMOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTanh());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;
  SimpleLSTMOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 10000;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 24;

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(g_simple_lstm_model_data), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 3);

  // some dummy input
  interpreter.input(0)->data.f[0] = 1.f;
  interpreter.input(0)->data.f[1] = 1.f;

  // initialize hidden and cell state
  for (int i = 0; i < 20; i++) {
    interpreter.input(1)->data.f[i] = 0.f;
    interpreter.input(2)->data.f[i] = 0.f;
  }

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(g_simple_lstm_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  SimpleLSTMOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 10000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  std::cout << "tensors alloated!" << "\n";

  // Check if the predicted output is within a small range of the
  // expected output
  float epsilon = 0.05f;
  constexpr int kNumTestValues = 4;
  float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};

  float expected_outputs[kNumTestValues] = {-0.1274f, -0.1704f, -0.2317f, -0.2682f};

  // TODO: why is the order of input and output tensors changed???
  // the tensors are now 1,0,2 instead of 0,1,2 as in pytorch for 0 -> input, 1 -> h, 2 -> c
  for (int i = 0; i < kNumTestValues; ++i) {
    interpreter.input(1)->data.f[0] = golden_inputs[i];
    interpreter.input(1)->data.f[1] = golden_inputs[i];

    for (int j = 0; j < 20; j++) {
      interpreter.input(0)->data.f[j] = 0.f;
      interpreter.input(2)->data.f[j] = 0.f;
    }

    TF_LITE_ENSURE_STATUS(interpreter.Invoke());

    // Retrieve and print all outputs
    const auto& output_indices = interpreter.outputs();
    for (size_t i = 0; i < output_indices.size(); ++i) {
        const auto* output_tensor = interpreter.output(i);
        const float* output_data = output_tensor->data.f;
        size_t num_elements = output_tensor->bytes / sizeof(float);

        std::cout << "Output " << i << ": ";
        for (size_t j = 0; j < num_elements; ++j) {
            std::cout << output_data[j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Interpreter invoked!" << "\n";
    float y_pred = interpreter.output(1)->data.f[0];
    std::cout << "Output retrieved!: " << y_pred << "\n";
    TFLITE_CHECK_LE(abs(expected_outputs[i] - y_pred), epsilon);
  }

  return kTfLiteOk;
}



int main(int argc, char* argv[]) {
  tflite::InitializeTarget();
  std::cout << "Starting Simple LSTM" << "\n";
  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  std::cout << "Starting LoadFloatModelAndPerformInference" << "\n";
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
