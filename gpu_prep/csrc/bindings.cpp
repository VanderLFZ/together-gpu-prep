#include <torch/extension.h>

#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Original CUDA FP32 forward-only interview teaching kernels";
  m.def("vector_add", &gpu_prep::vector_add_cuda, py::arg("x"), py::arg("y"));
  m.def("bias_relu", &gpu_prep::bias_relu_cuda, py::arg("x"), py::arg("bias"));
  m.def("row_sum", &gpu_prep::row_sum_cuda, py::arg("x"));
  m.def("row_max", &gpu_prep::row_max_cuda, py::arg("x"));
  m.def("softmax", &gpu_prep::softmax_cuda, py::arg("x"));
  m.def("transpose_naive", &gpu_prep::transpose_naive_cuda, py::arg("x"));
  m.def("transpose_tiled", &gpu_prep::transpose_tiled_cuda, py::arg("x"));
  m.def("matmul_naive", &gpu_prep::matmul_naive_cuda, py::arg("a"), py::arg("b"));
  m.def("matmul_tiled", &gpu_prep::matmul_tiled_cuda, py::arg("a"), py::arg("b"));
  m.def("layer_norm", &gpu_prep::layer_norm_cuda, py::arg("x"),
        py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
  m.def("rms_norm", &gpu_prep::rms_norm_cuda, py::arg("x"),
        py::arg("weight"), py::arg("eps") = 1e-5);
  m.def("attention", &gpu_prep::attention_cuda, py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("causal") = false,
        py::arg("query_start") = 0);
}
