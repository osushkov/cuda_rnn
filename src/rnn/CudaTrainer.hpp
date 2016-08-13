#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "RNNSpec.hpp"
#include "SliceBatch.hpp"

namespace rnn {

class CudaTrainer {
public:
  CudaTrainer(const RNNSpec &spec, unsigned maxTraceLength);
  ~CudaTrainer();

  void SetWeights(const vector<math::MatrixView> &weights);
  void GetWeights(vector<math::MatrixView> &outWeights);

  void Train(const vector<SliceBatch> &trace);

private:
  struct CudaTrainerImpl;
  uptr<CudaTrainerImpl> impl;
};
}
