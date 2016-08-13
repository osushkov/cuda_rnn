
#include "Util.hpp"
#include "Util.cuh"
#include <cuda_runtime.h>
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

void *util::AllocPinned(size_t bufSize) {
  void* result = nullptr;

  cudaError_t err = cudaHostAlloc(&result, bufSize, cudaHostAllocDefault);
  CheckError(err);
  assert(result != nullptr);

  return result;
}

void util::FreePinned(void *buf) {
  assert(buf != nullptr);
  cudaError_t err = cudaFreeHost(buf);
  CheckError(err);
}

CuMatrix util::AllocMatrix(unsigned rows, unsigned cols) {
  CuMatrix result;

  result.data = nullptr;
  result.rows = rows;
  result.cols = cols;

  size_t width = cols * sizeof(float);
  size_t height = rows;

  cudaError_t err = cudaMallocPitch(&(result.data), &(result.pitch), width, height);
  CheckError(err);
  assert(result.data != nullptr);

  return result;
}

void util::FreeMatrix(CuMatrix &m) {
  assert(m.data != nullptr);
  cudaError_t err = cudaFree(m.data);
  CheckError(err);
  m.data = nullptr;
}
