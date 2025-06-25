#include <Python.h>
#include <torch/library.h>

extern "C" {
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
  };
  return PyModule_Create(&module_def);
}
}  // extern "C"

TORCH_LIBRARY(hpc, m) {}
