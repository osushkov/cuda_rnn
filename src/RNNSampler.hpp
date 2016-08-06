#pragma once

#include "common/Common.hpp"
#include "rnn/RNN.hpp"
#include <vector>

class RNNSampler {
public:
  RNNSampler(unsigned letterDim);
  ~RNNSampler();

  vector<unsigned> SampleCharacters(rnn::RNN *network, unsigned numChars);

private:
  struct RNNSamplerImpl;
  uptr<RNNSamplerImpl> impl;
};
