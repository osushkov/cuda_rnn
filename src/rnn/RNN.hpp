
#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "RNNSpec.hpp"
#include "SliceBatch.hpp"

namespace rnn {

class RNN {
public:
  RNN(const RNNSpec &spec);
  virtual ~RNN();

  RNN &operator=(const RNN &other);

  void ClearMemory(void);
  EMatrix Process(const EMatrix &input);

  void Update(const vector<SliceBatch> &trace);
  void Refresh(void);

private:
  struct RNNImpl;
  uptr<RNNImpl> impl;
};
}
