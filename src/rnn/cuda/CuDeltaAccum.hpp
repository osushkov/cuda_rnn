#pragma once

#include "../LayerDef.hpp"
#include "Types.hpp"
#include "Util.hpp"
#include <cassert>
#include <vector>

using namespace std;

namespace rnn {
namespace cuda {

struct CuLayerAccum {
  unsigned layerId;
  int timestamp;

  unsigned samples;
  CuMatrix accumDelta;

  CuLayerAccum(unsigned layerId, int timestamp, unsigned deltaRows, unsigned deltaCols)
      : layerId(layerId), timestamp(timestamp), samples(0),
        accumDelta(util::AllocMatrix(deltaRows, deltaCols)) {}

  void Cleanup(void) { util::FreeMatrix(accumDelta); }
};

struct CuDeltaAccum {
  vector<CuLayerAccum> allDeltaAccum;

  CuDeltaAccum(unsigned maxBatchSize, unsigned maxTraceLength, const vector<LayerSpec> &layers);
  void Cleanup(void);

  CuLayerAccum *GetDelta(unsigned layerId, int timestamp);
  void Clear(void);
};
}
}
