#pragma once

#include "../LayerDef.hpp"
#include "Types.hpp"

namespace rnn {
namespace cuda {

enum class Task {
  LAYER_ACTIVATION,
  LAYER_SOFTMAX,
  PROPAGATE_DELTA,
  GRADIENT_INCREMENT,
  FILL_MATRIX,
  SCALE_MATRIX,
  TRANSPOSE_MATRIX,
  FORWARD_INCREMENT,
  COPY_MATRIX_D2H,
  COPY_MATRIX_H2D
};

struct LayerActivationData {
  ConnectionActivation layer;
  LayerActivation activation;

  LayerActivationData(ConnectionActivation layer, LayerActivation activation)
      : layer(layer), activation(activation) {}
};

struct LayerSoftmaxData {
  ConnectionActivation layer;

  LayerSoftmaxData(ConnectionActivation layer) : layer(layer) {}
};

struct PropagateDeltaData {
  LayerBatchDeltas nextDelta;
  CuMatrix transposedWeights;
  ConnectionActivation connection;
  LayerBatchDeltas outDelta;

  PropagateDeltaData(LayerBatchDeltas nextDelta, CuMatrix transposedWeights,
                     ConnectionActivation connection, LayerBatchDeltas outDelta)
      : nextDelta(nextDelta), transposedWeights(transposedWeights), connection(connection),
        outDelta(outDelta) {}
};

struct GradientIncrementData {
  LayerBatchDeltas layerDeltas;
  ConnectionActivation connection;
  CuMatrix outGradient;

  GradientIncrementData(LayerBatchDeltas layerDeltas, ConnectionActivation connection,
                        CuMatrix outGradient)
      : layerDeltas(layerDeltas), connection(connection), outGradient(outGradient) {}
};

struct FillMatrixData {
  CuMatrix target;
  float value;

  FillMatrixData(CuMatrix target, float value) : target(target), value(value) {}
};

struct ScaleMatrixData {
  CuMatrix target;
  float scale;

  ScaleMatrixData(CuMatrix target, float scale) : target(target), scale(scale) {}
};

struct TransposeMatrixData {
  CuMatrix src;
  CuMatrix dst;

  TransposeMatrixData(CuMatrix src, CuMatrix dst) : src(src), dst(dst) {}
};

union TaskData {
  LayerActivationData layerActivationData;
  LayerSoftmaxData layerSoftmaxData;
};
}
}
