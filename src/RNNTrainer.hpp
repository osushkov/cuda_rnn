#pragma once

#include "CharacterStream.hpp"
#include "common/Common.hpp"
#include "rnn/RNN.hpp"

class RNNTrainer {
public:
  RNNTrainer(unsigned miniTraceLength);
  ~RNNTrainer();

  uptr<rnn::RNN> TrainLanguageNetwork(CharacterStream &cStream, unsigned iters);

private:
  struct RNNTrainerImpl;
  uptr<RNNTrainerImpl> impl;
};
