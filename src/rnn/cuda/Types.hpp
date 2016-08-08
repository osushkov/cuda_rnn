
#pragma once

#include <cassert>
#include <cstddef>

namespace rnn {
namespace cuda {

struct CuMatrix {
  unsigned rows;
  unsigned cols;

  // Data pointers allocated with cudaMallocPitch. Logical size is (rows * cols).
  // Elements are stored in row major format (inner loop traverses across a given row).
  float *data;

  // The pitch of the rows of the weights matrix in bytes.
  size_t pitch;
};

struct SamplesBatch {
  unsigned batchSize; // equal to the number of rows in the matrix actually used.

  // each sample is a row vector in the input and targetOutput matrices.
  CuMatrix input;
  CuMatrix targetOutput;
};

struct ConnectionActivation {
  unsigned batchSize; // equal to the number of rows in the matrix actually used.

  CuMatrix activation;
  CuMatrix derivative;
};

struct LayerBatchDeltas {
  unsigned batchSize; // equal to the number of rows in the matrix actually used.
  CuMatrix delta;
};
}
}
