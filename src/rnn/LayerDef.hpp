#pragma once

#include <cassert>

namespace rnn {

enum class LayerActivation { TANH, LOGISTIC, RELU, LEAKY_RELU, ELU, LINEAR, SOFTMAX };

struct LayerConnection {
  unsigned srcLayerId;
  unsigned dstLayerId;

  int timeOffset; // should be 0 or 1

  LayerConnection(unsigned srcLayerId, unsigned dstLayerId, int timeOffset)
      : srcLayerId(srcLayerId), dstLayerId(dstLayerId), timeOffset(timeOffset) {
    assert(timeOffset == 0 || timeOffset == 1);
  }

  bool operator==(const LayerConnection &other) const {
    return this->srcLayerId == other.srcLayerId && this->dstLayerId == other.dstLayerId &&
           this->timeOffset == other.timeOffset;
  }
};

struct LayerSpec {
  unsigned uid; // must be >= 1, 0 is the "input" layer.
  unsigned numNodes;
  bool isOutput;

  LayerSpec(unsigned uid, unsigned numNodes, bool isOutput)
      : uid(uid), numNodes(numNodes), isOutput(isOutput) {
    assert(uid >= 1);
    assert(numNodes > 0);
  }
};
}
