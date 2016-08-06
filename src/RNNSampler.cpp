#include "RNNSampler.hpp"
#include <cassert>

struct RNNSampler::RNNSamplerImpl {
  unsigned letterDim;

  RNNSamplerImpl(unsigned letterDim) : letterDim(letterDim) { assert(letterDim > 0); }

  vector<unsigned> SampleCharacters(rnn::RNN *network, unsigned numChars) {
    vector<unsigned> result;
    result.reserve(numChars);

    network->ClearMemory();
    for (unsigned i = 0; i < numChars; i++) {
      unsigned sample = sampleChar(network, result);
      result.push_back(sample);
    }

    return result;
  }

  unsigned sampleChar(rnn::RNN *rnn, const vector<unsigned> &prevChars) {
    EVector input(letterDim);
    input.fill(0.0f);

    if (prevChars.size() > 0) {
      input(prevChars.back()) = 1.0f;
    }

    EVector pChar = rnn->Process(input);
    float r = math::UnitRand();

    for (int i = 0; i < pChar.rows(); i++) {
      r -= pChar(i);
      if (r < 0.0f) {
        return static_cast<unsigned>(i);
      }
    }

    return static_cast<unsigned>(rand() % pChar.rows());
  }
};

RNNSampler::RNNSampler(unsigned letterDim) : impl(new RNNSamplerImpl(letterDim)) {}

RNNSampler::~RNNSampler() = default;

vector<unsigned> RNNSampler::SampleCharacters(rnn::RNN *network, unsigned numChars) {
  return impl->SampleCharacters(network, numChars);
}
