#pragma once

#include "../../math/MatrixView.hpp"
#include "../LayerDef.hpp"
#include "Types.hpp"

namespace rnn {
namespace cuda {

enum class TaskType {
  LAYER_ACTIVATION,
  LAYER_SOFTMAX,
  PROPAGATE_DELTA,
  GRADIENT_INCREMENT,
  FILL_MATRIX,
  SCALE_MATRIX,
  TRANSPOSE_MATRIX,
  FORWARD_INCREMENT,
  COPY_MATRIX_D2H,
  COPY_MATRIX_H2D,
  COPY_MATRIX_D2D,
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

  FillMatrixData(CuMatrix target, float value) : target(target), value(value) {
    assert(target.data != nullptr);
  }
};

struct ScaleMatrixData {
  CuMatrix target;
  float scale;

  ScaleMatrixData(CuMatrix target, float scale) : target(target), scale(scale) {
    assert(target.data != nullptr);
  }
};

struct TransposeMatrixData {
  CuMatrix src;
  CuMatrix dst;

  TransposeMatrixData(CuMatrix src, CuMatrix dst) : src(src), dst(dst) {
    assert(dst.cols >= src.rows);
    assert(dst.rows >= src.cols);
    assert(src.data != nullptr && dst.data != nullptr);
  }
};

struct ForwardIncrementData {
  CuMatrix layerWeights;
  ConnectionActivation input;
  CuMatrix output;

  ForwardIncrementData(CuMatrix layerWeights, ConnectionActivation input, CuMatrix output)
      : layerWeights(layerWeights), input(input), output(output) {}
};

struct CopyMatrixD2HData {
  CuMatrix src;
  math::MatrixView dst;

  CopyMatrixD2HData(CuMatrix src, math::MatrixView dst) : src(src), dst(dst) {
    assert(dst.rows >= src.rows);
    assert(dst.cols >= src.cols);
    assert(src.data != nullptr && dst.data != nullptr);
  }
};

struct CopyMatrixH2DData {
  math::MatrixView src;
  CuMatrix dst;

  CopyMatrixH2DData(math::MatrixView src, CuMatrix dst) : src(src), dst(dst) {
    assert(dst.rows >= src.rows);
    assert(dst.cols >= src.cols);
    assert(src.data != nullptr && dst.data != nullptr);
  }
};

struct CopyMatrixD2DData {
  CuMatrix src;
  CuMatrix dst;

  CopyMatrixD2DData(CuMatrix src, CuMatrix dst) : src(src), dst(dst) {
    assert(dst.rows >= src.rows);
    assert(dst.cols >= src.cols);
    assert(src.data != nullptr && dst.data != nullptr);
  }
};

union TaskData {
  LayerActivationData layerActivationData;
  LayerSoftmaxData layerSoftmaxData;
  PropagateDeltaData propagateDeltaData;
  GradientIncrementData gradientIncrementData;
  FillMatrixData fillMatrixData;
  ScaleMatrixData scaleMatrixData;
  TransposeMatrixData transposeMatrixData;
  ForwardIncrementData forwardIncrementData;
  CopyMatrixD2HData copyMatrixD2HData;
  CopyMatrixH2DData copyMatrixH2DData;
  CopyMatrixD2DData copyMatrixD2DData;
};

struct Task {
  TaskType type;
  TaskData data;
};
}
}
