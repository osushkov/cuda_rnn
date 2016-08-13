
#include "CudaTrainer.hpp"
#include "cuda/CuLayer.hpp"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

struct CudaTrainer::CudaTrainerImpl {
  RNNSpec spec;
  unsigned maxTraceLength;

  vector<CuLayer> layers;

  CudaTrainerImpl(const RNNSpec &spec, unsigned maxTraceLength)
      : spec(spec), maxTraceLength(maxTraceLength) {
    assert(maxTraceLength > 0);
  }

  void SetWeights(const vector<math::MatrixView> &weights) {}
  void GetWeights(vector<math::MatrixView> &outWeights) {}

  void Train(const vector<SliceBatch> &trace) {}
};

CudaTrainer::CudaTrainer(const RNNSpec &spec, unsigned maxTraceLength)
    : impl(new CudaTrainerImpl(spec, maxTraceLength)) {}

CudaTrainer::~CudaTrainer() = default;

void CudaTrainer::SetWeights(const vector<math::MatrixView> &weights) { impl->SetWeights(weights); }

void CudaTrainer::GetWeights(vector<math::MatrixView> &outWeights) { impl->GetWeights(outWeights); }

void CudaTrainer::Train(const vector<SliceBatch> &trace) { impl->Train(trace); }
