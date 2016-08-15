
#include "CudaTrainer.hpp"
#include "../common/Common.hpp"
#include "../math/MatrixView.hpp"
#include "cuda/CuAdamState.hpp"
#include "cuda/CuDeltaAccum.hpp"
#include "cuda/CuGradientAccum.hpp"
#include "cuda/CuLayer.hpp"
#include "cuda/CuLayerMemory.hpp"
#include "cuda/TaskExecutor.hpp"
#include "cuda/Util.hpp"
#include <cassert>
#include <cstring>
#include <utility>

using namespace rnn;
using namespace rnn::cuda;

struct CudaTrainer::CudaTrainerImpl {
  RNNSpec spec;
  unsigned maxTraceLength;

  vector<CuLayer> layers;

  CuDeltaAccum deltaAccum;
  CuGradientAccum gradientAccum;
  CuLayerMemory layerMemory;
  CuAdamState adamState;

  TaskExecutor executor;
  vector<pair<math::MatrixView, math::MatrixView>> inputOutputStaging;

  CudaTrainerImpl(const RNNSpec &spec, unsigned maxTraceLength)
      : spec(spec), maxTraceLength(maxTraceLength), deltaAccum(spec, maxTraceLength),
        gradientAccum(spec), layerMemory(spec, maxTraceLength), adamState(spec) {
    assert(maxTraceLength > 0);
    adamState.Clear();

    for (unsigned i = 0; i < maxTraceLength; i++) {
      math::MatrixView inputStaging;
      inputStaging.rows = spec.maxBatchSize;
      inputStaging.cols = spec.numInputs;
      inputStaging.data =
          (float *)util::AllocPinned(inputStaging.rows * inputStaging.cols * sizeof(float));

      math::MatrixView outputStaging;
      outputStaging.rows = spec.maxBatchSize;
      outputStaging.cols = spec.numOutputs;
      outputStaging.data =
          (float *)util::AllocPinned(outputStaging.rows * outputStaging.cols * sizeof(float));

      inputOutputStaging.emplace_back(inputStaging, outputStaging);
    }
  }

  ~CudaTrainerImpl() {
    for (auto &layer : layers) {
      layer.Cleanup();
    }

    deltaAccum.Cleanup();
    gradientAccum.Cleanup();
    layerMemory.Cleanup();
    adamState.Cleanup();

    for (auto &staging : inputOutputStaging) {
      util::FreePinned(staging.first.data);
      util::FreePinned(staging.second.data);
    }
  }

  void SetWeights(const vector<math::MatrixView> &weights) {}
  void GetWeights(vector<math::MatrixView> &outWeights) {}

  void Train(const vector<SliceBatch> &trace) {
    assert(trace.size() <= maxTraceLength);
    assert(!trace.empty());
    assert(trace.front().batchInput.rows() == trace.front().batchOutput.rows());

    deltaAccum.Clear();
    gradientAccum.Clear();
    layerMemory.Clear();

    pushTraceToDevice(trace);

    unsigned batchSize = trace.front().batchInput.rows();
    for (int i = 0; i < static_cast<int>(trace.size()); i++) {
      forwardProp(i, batchSize);
    }

    for (int i = static_cast<int>(trace.size()) - 1; i >= 0; i--) {
      backProp(i, batchSize);
    }

    updateLayers(batchSize);
  }

  void pushTraceToDevice(const vector<SliceBatch> &trace) {
    for (unsigned i = 0; i < trace.size(); i++) {
      assert(trace[i].batchInput.cols() == inputOutputStaging[i].first.cols);
      assert(trace[i].batchInput.rows() <= inputOutputStaging[i].first.rows);
      assert(trace[i].batchOutput.cols() == inputOutputStaging[i].second.cols);
      assert(trace[i].batchOutput.rows() <= inputOutputStaging[i].second.rows);

      size_t inputSize = trace[i].batchInput.rows() * trace[i].batchInput.cols() * sizeof(float);
      memcpy(inputOutputStaging[i].first.data, trace[i].batchInput.data(), inputSize);

      size_t outputSize = trace[i].batchOutput.rows() * trace[i].batchOutput.cols() * sizeof(float);
      memcpy(inputOutputStaging[i].second.data, trace[i].batchOutput.data(), outputSize);

      CuTimeSlice *ts = layerMemory.GetTimeSlice(static_cast<int>(i));
      assert(ts != nullptr);

      executor.Execute(Task::CopyMatrixH2D(inputOutputStaging[i].second, ts->networkOutput));
      for (auto &cd : ts->connectionData) {
        if (cd.connection.srcLayerId == 0) {
          executor.Execute(Task::CopyMatrixH2D(inputOutputStaging[i].first, cd.activation));
        }
      }
    }
  }

  void forwardProp(int timestamp, unsigned batchSize) {
    for (auto &layer : layers) {
      assert(!layer.outgoing.empty());
      assert(!layer.incoming.empty());

      vector<CuConnectionMemoryData *> outData = getAllOutgoingConnections(layer, timestamp);

      // This should only be possible if all the outgoing connections are recurrent.
      // Currently, this is assumed to not be possible.
      assert(!outData.empty());

      CuConnectionMemoryData *targetOut = outData[0];
      assert(!targetOut->haveActivation);

      for (auto &in : layer.incoming) {
        if (in.first.timeOffset == 1 && timestamp == 0) {
          continue;
        }

        CuConnectionMemoryData *inData = getConnectionMemoryData(in.first, timestamp);
        assert(inData != nullptr && inData->haveActivation);

        ConnectionActivation activationIn(batchSize, inData->activation, inData->derivative);
        executor.Execute(Task::ForwardIncrement(in.second, activationIn, targetOut->activation));
      }

      ConnectionActivation outActivation(batchSize, targetOut->activation, targetOut->derivative);
      executor.Execute(Task::LayerActivation(outActivation, layer.activation));
      targetOut->haveActivation = true;

      for (unsigned i = 1; i < outData.size(); i++) {
        assert(!outData[i]->haveActivation);

        executor.Execute(Task::CopyMatrixD2D(targetOut->activation, outData[i]->activation));
        executor.Execute(Task::CopyMatrixD2D(targetOut->derivative, outData[i]->derivative));
        outData[i]->haveActivation = true;
      }
    }
  }

  vector<CuConnectionMemoryData *> getAllOutgoingConnections(const CuLayer &layer, int timestamp) {
    vector<CuConnectionMemoryData *> result;
    for (auto &conn : layer.outgoing) {
      CuConnectionMemoryData *cmd = getConnectionMemoryData(conn, timestamp + conn.timeOffset);
      if (cmd != nullptr) {
        result.push_back(cmd);
      }
    }

    return result;
  }

  CuConnectionMemoryData *getConnectionMemoryData(const LayerConnection &conn, int timestamp) {
    CuTimeSlice *ts = layerMemory.GetTimeSlice(timestamp);
    if (ts == nullptr) {
      return nullptr;
    }

    return ts->GetConnectionData(conn);
  }

  void backProp(int timestamp, unsigned batchSize) {}

  void recursiveBackprop(const CuLayer &layer, int timestamp, unsigned batchSize) {
    CuLayerAccum *layerDelta = deltaAccum.GetDelta(layer.layerId, timestamp);
    assert(layerDelta != nullptr);

    assert(layerDelta->samples > 0);
    float deltaScale = 1.0f / static_cast<float>(layerDelta->samples);
    executor.Execute(Task::ScaleMatrix(layerDelta->accumDelta, deltaScale));

    LayerBatchDeltas batchDelta(batchSize, layerDelta->accumDelta);

    CuTimeSlice *slice = layerMemory.GetTimeSlice(timestamp);
    assert(slice != nullptr);

    for (const auto &connection : layer.incoming) {
      if (connection.first.timeOffset == 1 && timestamp == 0) {
        continue;
      }

      CuConnectionMemoryData *connData = slice->GetConnectionData(connection.first);
      assert(connData != nullptr && connData->haveActivation);

      CuConnectionAccum *connAccum = gradientAccum.GetConnection(connection.first);
      assert(connAccum != nullptr);

      ConnectionActivation activationIn(batchSize, connData->activation, connData->derivative);
      executor.Execute(Task::GradientIncrement(batchDelta, activationIn, connAccum->accumGradient));
      connAccum->samples++;

      if (connection.first.srcLayerId != 0) { // The source is another layer from the srcSlice.
        // CuLayer *srcLayer = findLayer(connection.first.srcLayerId);

        // TODO: store the weight transpose in the CuLayer to save compute time.
        // TODO: debug the backprop in CPU RNN because I think its wrong. Its taking the
        // incorrect slice.

        // // Now increment the delta for the src layer.
        // int nRows = connection.second.rows();
        // int nCols = connection.second.cols() - 1;
        // EMatrix noBiasWeights = connection.second.bottomLeftCorner(nRows, nCols);
        // EMatrix srcDelta = noBiasWeights.transpose() * delta;
        // assert(srcLayer.numNodes == srcDelta.rows());
        //
        // componentScale(srcDelta, cmd->derivative);
        // LayerAccum &deltaAccum =
        //     bpContext.deltaAccum.IncrementDelta(srcLayer.layerId, srcTimestamp, srcDelta);
        //
        // if (connection.first.timeOffset == 0) {
        //   recursiveBackprop(srcLayer, timestamp, batchSize);
        // }
      }
    }
  }

  CuLayer *findLayer(unsigned layerId) {
    auto r = find_if(layers, [layerId](const CuLayer &l) { return l.layerId == layerId; });
    return r.valid() ? &(r.val()) : nullptr;
  }

  void updateLayers(unsigned batchSize) {}
};

CudaTrainer::CudaTrainer(const RNNSpec &spec, unsigned maxTraceLength)
    : impl(new CudaTrainerImpl(spec, maxTraceLength)) {}

CudaTrainer::~CudaTrainer() = default;

void CudaTrainer::SetWeights(const vector<math::MatrixView> &weights) { impl->SetWeights(weights); }

void CudaTrainer::GetWeights(vector<math::MatrixView> &outWeights) { impl->GetWeights(outWeights); }

void CudaTrainer::Train(const vector<SliceBatch> &trace) { impl->Train(trace); }
