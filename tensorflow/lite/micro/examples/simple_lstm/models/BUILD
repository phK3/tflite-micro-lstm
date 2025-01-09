load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = [
        "simple_lstm.tflite",
    ],
    visibility = [
        "//codegen/examples/simple_lstm:__subpackages__",
        "//tensorflow/lite/micro/examples/simple_lstm:__subpackages__",
    ],
)

generate_cc_arrays(
    name = "generated_simple_lstm_cc",
    src = "simple_lstm.tflite",
    out = "simple_lstm_data.cc",
)

generate_cc_arrays(
    name = "generated_simple_lstm_hdr",
    src = "simple_lstm.tflite",
    out = "simple_lstm_data.h",
)
