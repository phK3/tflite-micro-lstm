<!-- mdformat off(b/169948621#comment2) -->

# Simple LSTM Example

Demonstrates how to execute a simple LSTM model with `tflite-micro`.

## Run the Example

To run the tests on the example, just navigate to the `tflite-micro-lstm` root directory and execute:
```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_simple_lstm_test
```

## Using a Different Model

As long as the order of inputs and outputs isn't changed, different models can be used by simply substituting `simple_lstm.tflite` with the `.tflite` file of another model.

Maybe `kTensorArenaSize` needs to be adjusted, if the model is larger or we need to register additional operations in `SimpleLSTMOpResolver` if the new model requires them. 

The current model is just a dummy PyTorch model that was converted to tflite using the notebook under https://colab.research.google.com/drive/1O_CAxWYdJm6TkHUlrO1mnCZo8vnlAJvJ?usp=sharing 