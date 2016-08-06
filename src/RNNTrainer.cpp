
#include "RNNTrainer.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace rnn;

static constexpr unsigned TRAINING_SIZE = 100 * 1000 * 1000;
static constexpr unsigned BATCH_SIZE = 16;

struct RNNTrainer::RNNTrainerImpl {
  unsigned traceLength;

  RNNTrainerImpl(unsigned traceLength) : traceLength(traceLength) {}

  uptr<RNN> TrainLanguageNetwork(CharacterStream &cStream, unsigned iters) {
    uptr<RNN> network = createNewNetwork(cStream.VectorDimension(), cStream.VectorDimension());

    vector<math::OneHotVector> letters = cStream.ReadCharacters(TRAINING_SIZE);
    for (unsigned i = 0; i < iters; i++) {
      if (i % 100 == 0) {
        cout << i << "/" << iters << endl;
      }

      network->Update(makeBatch(letters, BATCH_SIZE));
    }
    network->Refresh();

    return move(network);
  }

  vector<SliceBatch> makeBatch(const vector<math::OneHotVector> &trainingData, unsigned batchSize) {
    assert(trainingData.size() > traceLength);

    unsigned dim = trainingData.front().dim;

    vector<SliceBatch> result;
    result.reserve(traceLength);

    vector<unsigned> indices = createTraceStartIndices(trainingData.size(), batchSize);
    for (unsigned i = 0; i < traceLength; i++) {
      EMatrix input(dim, batchSize);
      EMatrix output(dim, batchSize);

      for (unsigned j = 0; j < batchSize; j++) {
        assert(trainingData[indices[j]].dim == dim);
        assert(trainingData[indices[j] + 1].dim == dim);

        input.col(j) = trainingData[indices[j]].DenseVector();
        output.col(j) = trainingData[indices[j] + 1].DenseVector();
        indices[j]++;
      }

      result.emplace_back(input, output);
    }

    return result;
  }

  vector<unsigned> createTraceStartIndices(unsigned dataLength, unsigned batchSize) {
    vector<unsigned> indices;
    for (unsigned i = 0; i < batchSize; i++) {
      indices.push_back(rand() % (dataLength - traceLength));
    }
    return indices;
  }

  uptr<RNN> createNewNetwork(unsigned inputSize, unsigned outputSize) {
    RNNSpec spec;

    spec.numInputs = inputSize;
    spec.numOutputs = outputSize;
    spec.hiddenActivation = LayerActivation::ELU;
    spec.outputActivation = LayerActivation::SOFTMAX;
    spec.nodeActivationRate = 1.0f;

    // Connect layer 1 to the input.
    spec.connections.emplace_back(0, 1, 0);

    // Connection layer 1 to layer 2, layer 2 to the output layer.
    spec.connections.emplace_back(1, 2, 0);
    spec.connections.emplace_back(2, 3, 0);

    // Recurrent self-connections for layers 1 and 2.
    spec.connections.emplace_back(1, 1, 1);
    spec.connections.emplace_back(2, 2, 1);

    // 2 layers, 1 hidden.
    spec.layers.emplace_back(1, 64, false);
    spec.layers.emplace_back(2, 128, false);
    spec.layers.emplace_back(3, outputSize, true);

    return make_unique<RNN>(spec);
  }
};

RNNTrainer::RNNTrainer(unsigned traceLength) : impl(new RNNTrainerImpl(traceLength)) {}

RNNTrainer::~RNNTrainer() = default;

uptr<RNN> RNNTrainer::TrainLanguageNetwork(CharacterStream &cStream, unsigned iters) {
  return impl->TrainLanguageNetwork(cStream, iters);
}
