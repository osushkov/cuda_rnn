#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace AdamKernel {

void UpdateMomentumAndRMS(CuMatrix gradient, CuMatrix momentum, CuMatrix rms,
                          float beta1, float beta2, cudaStream_t stream);

void UpdateWeightsWithAdam(CuMatrix weights, CuMatrix momentum, CuMatrix rms,
                           float beta1, float beta2, float lr, float epsilon,
                           cudaStream_t stream);
}
}
}
