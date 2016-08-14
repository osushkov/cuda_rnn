
#include "CudaTrainer.hpp"
#include "cuda/CuAdamState.hpp"
#include "cuda/CuDeltaAccum.hpp"
#include "cuda/CuGradientAccum.hpp"
#include "cuda/CuLayer.hpp"
#include "cuda/CuLayerMemory.hpp"
#include "cuda/TaskExecutor.hpp"
#include <cassert>

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

  CudaTrainerImpl(const RNNSpec &spec, unsigned maxTraceLength)
      : spec(spec), maxTraceLength(maxTraceLength), deltaAccum(spec, maxTraceLength),
        gradientAccum(spec), layerMemory(spec, maxTraceLength), adamState(spec) {
    assert(maxTraceLength > 0);
    adamState.Clear();
  }

  ~CudaTrainerImpl() {
    for (auto &layer : layers) {
      layer.Cleanup();
    }

    deltaAccum.Cleanup();
    gradientAccum.Cleanup();
    layerMemory.Cleanup();
    adamState.Cleanup();
  }

  void SetWeights(const vector<math::MatrixView> &weights) {}
  void GetWeights(vector<math::MatrixView> &outWeights) {}

  void Train(const vector<SliceBatch> &trace) {
    assert(trace.size() <= maxTraceLength);

    deltaAccum.Clear();
    gradientAccum.Clear();
    layerMemory.Clear();

    unsigned batchSize = 16; // TODO: derive this from the trace data.
    pushTraceToDevice(trace);
    for (int i = 0; i < static_cast<int>(trace.size()); i++) {
      forwardProp(i, batchSize);
    }

    for (int i = static_cast<int>(trace.size()) - 1; i >= 0; i--) {
      backProp(i, batchSize);
    }

    updateLayers(batchSize);
  }

  void pushTraceToDevice(const vector<SliceBatch> &trace) {}

  void forwardProp(int timestamp, unsigned batchSize) {
    for (auto &layer : layers) {
      assert(!layer.outgoing.empty());
      assert(!layer.incoming.empty());

      vector<CuConnectionMemoryData *> outData = getAllOutgoingConnections(layer, timestamp);

      // This should only be possible if all the outgoing connections are recurrent.
      // Currently, this is assumed to be not possible.
      assert(!outData.empty());

      CuConnectionMemoryData *targetOut = outData[0];
      assert(!targetOut->haveActivation);

      for (auto &in : layer.incoming) {
        if (in.first.timeOffset == 1 && timestamp == 0) {
          continue;
        }

        CuConnectionMemoryData *inData = getConnectionMemoryData(in.first, timestamp);
        assert(inData != nullptr && inData->haveActivation);

        ConnectionActivation connectionActivation(batchSize, inData->activation,
                                                  inData->derivative);
        executor.Execute(
            Task::ForwardIncrement(in.second, connectionActivation, targetOut->activation));
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

  void updateLayers(unsigned batchSize) {}
};

CudaTrainer::CudaTrainer(const RNNSpec &spec, unsigned maxTraceLength)
    : impl(new CudaTrainerImpl(spec, maxTraceLength)) {}

CudaTrainer::~CudaTrainer() = default;

void CudaTrainer::SetWeights(const vector<math::MatrixView> &weights) { impl->SetWeights(weights); }

void CudaTrainer::GetWeights(vector<math::MatrixView> &outWeights) { impl->GetWeights(outWeights); }

void CudaTrainer::Train(const vector<SliceBatch> &trace) { impl->Train(trace); }
