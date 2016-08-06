
#pragma once

#include "../common/Common.hpp"
#include "LayerDef.hpp"
#include <vector>

namespace rnn {

struct RNNSpec {
  unsigned numInputs;
  unsigned numOutputs;
  vector<LayerSpec> layers;
  vector<LayerConnection> connections;

  LayerActivation hiddenActivation;
  LayerActivation outputActivation;

  float nodeActivationRate; // for dropout regularization.
};
}
