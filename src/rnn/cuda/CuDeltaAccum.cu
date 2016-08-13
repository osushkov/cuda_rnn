
#include "CuDeltaAccum.hpp"
#include "MatrixFillKernel.cuh"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuDeltaAccum::CuDeltaAccum(unsigned maxBatchSize, unsigned maxTraceLength,
                           const vector<LayerSpec> &layers) {

  assert(maxBatchSize > 0 && maxTraceLength > 0);
  assert(layers.size() > 0);

  allDeltaAccum.reserve(maxTraceLength * layers.size());
  for (int timestamp = 0; timestamp < maxTraceLength; timestamp++) {
    for (const auto &layer : layers) {
      allDeltaAccum.emplace_back(layer.uid, timestamp, maxBatchSize, layer.numNodes);
    }
  }
}

void Cleanup(void);

CuLayerAccum *CuDeltaAccum::GetDelta(unsigned layerId, int timestamp) {
  for (auto &da : allDeltaAccum) {
    if (da.layerId == layerId && da.timestamp == timestamp) {
      return &da;
    }
  }

  assert(false);
  return nullptr;
}

void CuDeltaAccum::Clear(void) {
  for (auto &da : allDeltaAccum) {
    MatrixFillKernel::Apply(da.accumDelta, 0.0f, 0);
  }
}
