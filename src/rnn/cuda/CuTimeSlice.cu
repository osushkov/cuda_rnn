
#include "CuTimeSlice.hpp"
#include "kernels/MatrixFillKernel.cuh"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuTimeSlice::CuTimeSlice(const RNNSpec &spec, int timestamp)
    : timestamp(timestamp), networkInput(util::AllocMatrix(spec.maxBatchSize, spec.numInputs)),
      networkOutput(util::AllocMatrix(spec.maxBatchSize, spec.numOutputs)) {

  assert(timestamp >= 0);
  for (const auto &connection : spec.connections) {
    unsigned connectionCols = spec.LayerSize(connection.srcLayerId) + 1;
    connectionData.emplace_back(connection, spec.maxBatchSize, connectionCols);
  }
}

void CuTimeSlice::Cleanup(void) {
  util::FreeMatrix(networkInput);
  util::FreeMatrix(networkOutput);

  for (auto &cd : connectionData) {
    cd.Cleanup();
  }
}

CuConnectionMemoryData *CuTimeSlice::GetConnectionData(const LayerConnection &connection) {
  for (auto &cmd : connectionData) {
    if (cmd.connection == connection) {
      return &cmd;
    }
  }

  assert(false);
  return nullptr;
}

void CuTimeSlice::Clear(void) {
  MatrixFillKernel::Apply(networkInput, 0.0f, 0);
  MatrixFillKernel::Apply(networkOutput, 0.0f, 0);

  for (auto &cd : connectionData) {
    MatrixFillKernel::Apply(cd.activation, 0.0f, 0);
    MatrixFillKernel::Apply(cd.derivative, 0.0f, 0);
  }
}
