
#include "CudaTrainer.hpp"

using namespace rnn;

struct CudaTrainer::CudaTrainerImpl {
  RNNSpec spec;

  CudaTrainerImpl(const RNNSpec &spec) : spec(spec) {}

  void SetWeights(const vector<math::MatrixView> &weights) {}
  void GetWeights(vector<math::MatrixView> &outWeights) {}

  void Train(const vector<SliceBatch> &trace) {}
};

CudaTrainer::CudaTrainer(const RNNSpec &spec) : impl(new CudaTrainerImpl(spec)) {}
CudaTrainer::~CudaTrainer() = default;

void CudaTrainer::SetWeights(const vector<math::MatrixView> &weights) { impl->SetWeights(weights); }

void CudaTrainer::GetWeights(vector<math::MatrixView> &outWeights) { impl->GetWeights(outWeights); }

void CudaTrainer::Train(const vector<SliceBatch> &trace) { impl->Train(trace); }
