#pragma once

#include "../LayerDef.hpp"
#include "../RNNSpec.hpp"
#include "Types.hpp"
#include <utility>
#include <vector>

using namespace std;

namespace rnn {
namespace cuda {

struct CuLayer {
  unsigned layerId;
  LayerActivation activation;

  unsigned numNodes;
  bool isOutput;

  // Weights for incoming connections from other layers.
  vector<pair<LayerConnection, CuMatrix>> incoming;
  vector<LayerConnection> outgoing;

  CuLayer(const RNNSpec &nnSpec, const LayerSpec &layerSpec);
  void Cleanup(void);
};
}
}
