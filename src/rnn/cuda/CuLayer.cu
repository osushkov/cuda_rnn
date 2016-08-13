
#include "CuLayer.hpp"
#include "Util.hpp"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuLayer::CuLayer(const RNNSpec &nnSpec, const LayerSpec &layerSpec)
    : layerId(layerSpec.uid),
      activation(layerSpec.isOutput ? nnSpec.outputActivation : nnSpec.hiddenActivation),
      numNodes(layerSpec.numNodes), isOutput(layerSpec.isOutput) {

  assert(layerSpec.uid != 0); // 0 is reserved for the input.
  if (isOutput) {
    assert(numNodes == nnSpec.numOutputs);
  }

  for (const auto &lc : nnSpec.connections) {
    assert(lc.timeOffset == 0 || lc.timeOffset == 1);
    assert(lc.dstLayerId != 0);

    if (lc.dstLayerId == layerId) {
      // +1 accounts for the bias.
      unsigned inputSize = nnSpec.LayerSize(lc.srcLayerId) + 1;
      CuMatrix weightsMatrix = util::AllocMatrix(numNodes, inputSize);
      incoming.emplace_back(lc, weightsMatrix);
    }

    if (lc.srcLayerId == layerId) {
      outgoing.push_back(lc);
    }
  }
}

void CuLayer::Cleanup(void) {
  for (auto& ic : incoming) {
    util::FreeMatrix(ic.second);
  }
}
