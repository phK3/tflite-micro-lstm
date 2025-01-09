SIMPLE_LSTM_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/simple_lstm/simple_lstm_test.cc

SIMPLE_LSTM_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/simple_lstm/simple_lstm_test.cc

SIMPLE_LSTM_HDRS :=

SIMPLE_LSTM_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/simple_lstm/models/simple_lstm.tflite

SIMPLE_LSTM_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/simple_lstm/models/simple_lstm_data.cc

SIMPLE_LSTM_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/simple_lstm/models/simple_lstm_data.h

# Tests loading and running the simple_lstm model.
$(eval $(call microlite_test,simple_lstm_test,\
$(SIMPLE_LSTM_TEST_SRCS),,$(SIMPLE_LSTM_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,simple_lstm,\
$(SIMPLE_LSTM_SRCS),,$(SIMPLE_LSTM_GENERATOR_INPUTS)))

# Add sources and headers generated from $(SIMPLE_LSTM_GENERATOR_INPUTS).
SIMPLE_LSTM_SRCS += $(SIMPLE_LSTM_GENERATED_SRCS)
SIMPLE_LSTM_HDRS += $(SIMPLE_LSTM_GENERATED_HDRS)

list_simple_lstm_example_sources:
	@echo $(SIMPLE_LSTM_SRCS)

list_simple_lstm_example_headers:
	@echo $(SIMPLE_LSTM_HDRS)
