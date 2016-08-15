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

  LayerActivationData() = default;
  LayerActivationData(ConnectionActivation layer, LayerActivation activation)
      : layer(layer), activation(activation) {}
};

struct LayerSoftmaxData {
  ConnectionActivation layer;

  LayerSoftmaxData() = default;
  LayerSoftmaxData(ConnectionActivation layer) : layer(layer) {}
};

struct PropagateDeltaData {
  LayerBatchDeltas nextDelta;
  CuMatrix transposedWeights;
  ConnectionActivation connection;
  LayerBatchDeltas outDelta;

  PropagateDeltaData() = default;
  PropagateDeltaData(LayerBatchDeltas nextDelta, CuMatrix transposedWeights,
                     ConnectionActivation connection, LayerBatchDeltas outDelta)
      : nextDelta(nextDelta), transposedWeights(transposedWeights), connection(connection),
        outDelta(outDelta) {}
};

struct GradientIncrementData {
  LayerBatchDeltas layerDeltas;
  ConnectionActivation connection;
  CuMatrix outGradient;

  GradientIncrementData() = default;
  GradientIncrementData(LayerBatchDeltas layerDeltas, ConnectionActivation connection,
                        CuMatrix outGradient)
      : layerDeltas(layerDeltas), connection(connection), outGradient(outGradient) {}
};

struct FillMatrixData {
  CuMatrix target;
  float value;

  FillMatrixData() = default;
  FillMatrixData(CuMatrix target, float value) : target(target), value(value) {
    assert(target.data != nullptr);
  }
};

struct ScaleMatrixData {
  CuMatrix target;
  float scale;

  ScaleMatrixData() = default;
  ScaleMatrixData(CuMatrix target, float scale) : target(target), scale(scale) {
    assert(target.data != nullptr);
  }
};

struct TransposeMatrixData {
  CuMatrix src;
  CuMatrix dst;

  TransposeMatrixData() = default;
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

  ForwardIncrementData() = default;
  ForwardIncrementData(CuMatrix layerWeights, ConnectionActivation input, CuMatrix output)
      : layerWeights(layerWeights), input(input), output(output) {}
};

struct CopyMatrixD2HData {
  CuMatrix src;
  math::MatrixView dst;

  CopyMatrixD2HData() = default;
  CopyMatrixD2HData(CuMatrix src, math::MatrixView dst) : src(src), dst(dst) {
    assert(dst.rows >= src.rows);
    assert(dst.cols >= src.cols);
    assert(src.data != nullptr && dst.data != nullptr);
  }
};

struct CopyMatrixH2DData {
  math::MatrixView src;
  CuMatrix dst;

  CopyMatrixH2DData() = default;
  CopyMatrixH2DData(math::MatrixView src, CuMatrix dst) : src(src), dst(dst) {
    assert(dst.rows >= src.rows);
    assert(dst.cols >= src.cols);
    assert(src.data != nullptr && dst.data != nullptr);
  }
};

struct CopyMatrixD2DData {
  CuMatrix src;
  CuMatrix dst;

  CopyMatrixD2DData() = default;
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

  static Task LayerActivation(ConnectionActivation layer, LayerActivation activation) {
    Task task;
    task.type = TaskType::LAYER_ACTIVATION;
    task.data.layerActivationData = LayerActivationData(layer, activation);
    return task;
  }

  static Task LayerSoftmax(ConnectionActivation layer) {
    Task task;
    task.type = TaskType::LAYER_SOFTMAX;
    task.data.layerSoftmaxData = LayerSoftmaxData(layer);
    return task;
  }

  static Task PropagateDelta(LayerBatchDeltas nextDelta, CuMatrix transposedWeights,
                             ConnectionActivation connection, LayerBatchDeltas outDelta) {
    Task task;
    task.type = TaskType::PROPAGATE_DELTA;
    task.data.propagateDeltaData =
        PropagateDeltaData(nextDelta, transposedWeights, connection, outDelta);
    return task;
  }

  static Task GradientIncrement(LayerBatchDeltas layerDeltas, ConnectionActivation connection,
                                CuMatrix outGradient) {
    Task task;
    task.type = TaskType::GRADIENT_INCREMENT;
    task.data.gradientIncrementData = GradientIncrementData(layerDeltas, connection, outGradient);
    return task;
  }

  static Task FillMatrix(CuMatrix target, float value) {
    Task task;
    task.type = TaskType::FILL_MATRIX;
    task.data.fillMatrixData = FillMatrixData(target, value);
    return task;
  }

  static Task ScaleMatrix(CuMatrix target, float scale) {
    Task task;
    task.type = TaskType::SCALE_MATRIX;
    task.data.scaleMatrixData = ScaleMatrixData(target, scale);
    return task;
  }

  static Task ForwardIncrement(CuMatrix layerWeights, ConnectionActivation input, CuMatrix output) {
    Task task;
    task.type = TaskType::FORWARD_INCREMENT;
    task.data.forwardIncrementData = ForwardIncrementData(layerWeights, input, output);
    return task;
  }

  static Task CopyMatrixD2D(CuMatrix src, CuMatrix dst) {
    Task task;
    task.type = TaskType::COPY_MATRIX_D2D;
    task.data.copyMatrixD2DData = CopyMatrixD2DData(src, dst);
    return task;
  }
};
}
}
