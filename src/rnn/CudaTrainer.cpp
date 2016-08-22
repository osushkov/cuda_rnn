
#include "CudaTrainer.hpp"
#include "../common/Common.hpp"
#include "../common/Semaphore.hpp"
#include "../math/MatrixView.hpp"
#include "cuda/CuAdamState.hpp"
#include "cuda/CuDeltaAccum.hpp"
#include "cuda/CuGradientAccum.hpp"
#include "cuda/CuLayer.hpp"
#include "cuda/CuLayerMemory.hpp"
#include "cuda/TaskExecutor.hpp"
#include "cuda/Util.hpp"
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

using namespace rnn;
using namespace rnn::cuda;

constexpr float ADAM_BETA1 = 0.9f;
constexpr float ADAM_BETA2 = 0.999f;
constexpr float ADAM_LR = 0.001f;
constexpr float ADAM_EPSILON = 10e-8;

enum class TrainTask {
  NONE,
  EXIT,
  CLEAR_BUFFERS,
  FORWARDPROP,
  BACKPROP_DELTA,
  COMPUTE_AND_UPDATE_GRADIENTS,
};

struct CudaTrainer::CudaTrainerImpl {
  RNNSpec spec;
  unsigned maxTraceLength;

  vector<CuLayer> layers;

  CuDeltaAccum deltaAccum;
  CuGradientAccum gradientAccum;
  CuLayerMemory layerMemory;
  CuAdamState adamState;

  TaskExecutor defaultExecutor;
  vector<pair<math::MatrixView, math::MatrixView>> inputOutputStaging;
  vector<TargetOutput> traceTargets;

  mutex m; // controls access to tasks for the workers.
  condition_variable cv;
  vector<thread> workers;

  TrainTask currentWorkerTask;
  unsigned curBatchSize;
  unsigned curTraceLength;
  Semaphore taskSem;

  vector<TrainTask> taskList = {
      TrainTask::CLEAR_BUFFERS, TrainTask::FORWARDPROP, TrainTask::BACKPROP_DELTA,
      TrainTask::COMPUTE_AND_UPDATE_GRADIENTS,
  };

  CudaTrainerImpl(const RNNSpec &spec)
      : spec(spec), maxTraceLength(spec.maxTraceLength), deltaAccum(spec, maxTraceLength),
        gradientAccum(spec), layerMemory(spec, maxTraceLength), adamState(spec) {
    assert(maxTraceLength > 0);

    deltaAccum.Clear();
    gradientAccum.Clear();
    layerMemory.Clear();
    adamState.Clear();

    for (const auto &layerSpec : spec.layers) {
      layers.emplace_back(spec, layerSpec);
    }

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

      traceTargets.emplace_back(spec.maxBatchSize,
                                util::AllocMatrix(spec.maxBatchSize, spec.numOutputs));
    }

    createWorkers(4);
  }

  ~CudaTrainerImpl() {
    currentWorkerTask = TrainTask::EXIT;
    cv.notify_all();
    for (auto &w : workers) {
      w.join();
    }

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

    for (auto &tt : traceTargets) {
      util::FreeMatrix(tt.value);
    }
  }

  // TODO: SetWeights and GetWeights can share a whole bunch of code in a separate function, instead
  // of the current copy-paste.
  void SetWeights(const vector<pair<LayerConnection, math::MatrixView>> &inWeights) {
    // Don't care about speed here really, so we can skip staging memory.
    for (const auto &w : inWeights) {
      CuLayer *layer = findLayer(w.first.dstLayerId);
      assert(layer != nullptr);

      bool found = false;
      for (auto &li : layer->incoming) {
        if (li.first == w.first) {
          assert(li.second.weights.rows == w.second.rows);
          assert(li.second.weights.cols == w.second.cols);

          defaultExecutor.Execute(Task::CopyMatrixH2D(w.second, li.second.weights));
          defaultExecutor.Execute(Task::TransposeMatrix(li.second.weights, li.second.weightsT));
          found = true;
          break;
        }
      }
      assert(found);
    }
  }

  void GetWeights(vector<pair<LayerConnection, math::MatrixView>> &outWeights) {
    // Don't care about speed here really, so we can skip staging memory.
    for (const auto &w : outWeights) {
      CuLayer *layer = findLayer(w.first.dstLayerId);
      assert(layer != nullptr);

      bool found = false;
      for (auto &li : layer->incoming) {
        if (li.first == w.first) {
          assert(li.second.weights.rows == w.second.rows);
          assert(li.second.weights.cols == w.second.cols);

          defaultExecutor.Execute(Task::CopyMatrixD2H(li.second.weights, w.second));
          found = true;
          break;
        }
      }
      assert(found);
    }
  }

  void Train(const vector<SliceBatch> &trace) {
    assert(trace.size() <= maxTraceLength);
    assert(!trace.empty());
    assert(trace.front().batchInput.rows() == trace.front().batchOutput.rows());

    {
      std::lock_guard<std::mutex> lk(m);
      curBatchSize = trace.front().batchInput.rows();
      curTraceLength = trace.size();
    }

    pushTraceToDevice(trace);

    for (TrainTask task : taskList) {
      {
        std::lock_guard<std::mutex> lk(m);
        currentWorkerTask = task;
      }

      cv.notify_all();

      for (unsigned i = 0; i < workers.size(); i++) {
        taskSem.wait();
      }
      util::CudaSynchronize();
    }

    {
      std::lock_guard<std::mutex> lk(m);
      currentWorkerTask = TrainTask::NONE;
    }
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
    }
  }

  void createWorkers(unsigned numWorkers) {
    assert(numWorkers > 0);
    currentWorkerTask = TrainTask::NONE;

    for (unsigned workerIdx = 0; workerIdx < numWorkers; workerIdx++) {
      workers.emplace_back([this, workerIdx] {
        TaskExecutor executor;
        TrainTask prevTask = TrainTask::NONE;

        while (true) {
          std::unique_lock<std::mutex> lk(m);

          while (currentWorkerTask == TrainTask::NONE || currentWorkerTask == prevTask) {
            cv.wait(lk, [this, prevTask]() {
              return currentWorkerTask != TrainTask::NONE && currentWorkerTask != prevTask;
            });
          }

          assert(currentWorkerTask != TrainTask::NONE);
          prevTask = currentWorkerTask;
          lk.unlock();

          switch (prevTask) {
          case TrainTask::EXIT:
            return;
          case TrainTask::CLEAR_BUFFERS:
            workerClearBuffers(executor, workerIdx);
            break;
          case TrainTask::FORWARDPROP:
            workerForwardprop(executor, workerIdx);
            break;
          case TrainTask::BACKPROP_DELTA:
            workerBackpropDelta(executor, workerIdx);
            break;
          case TrainTask::COMPUTE_AND_UPDATE_GRADIENTS:
            workerComputeAndUpdateGradients(executor, workerIdx);
            break;
          default:
            assert(false);
          }

          taskSem.notify();
        }
      });
    }
  }

  void workerClearBuffers(TaskExecutor &executor, unsigned workerIdx) {
    unsigned skip = workers.size();

    for (unsigned i = workerIdx; i < deltaAccum.allDeltaAccum.size(); i += skip) {
      deltaAccum.allDeltaAccum[i].samples = 0;
      executor.Execute(Task::FillMatrix(deltaAccum.allDeltaAccum[i].accumDelta, 0.0f));
    }

    for (unsigned i = workerIdx; i < gradientAccum.allWeightsAccum.size(); i += skip) {
      gradientAccum.allWeightsAccum[i].samples = 0;
      // executor.Execute(Task::FillMatrix(gradientAccum.allWeightsAccum[i].accumGradient, 0.0f));
    }

    for (unsigned i = workerIdx; i < spec.maxTraceLength; i += skip) {
      CuTimeSlice *ts = layerMemory.GetTimeSlice(i);
      if (ts != nullptr) {
        ts->networkOutput.haveActivation = false;
        executor.Execute(Task::FillMatrix(ts->networkOutput.activation, 0.0f));
        // executor.Execute(Task::FillMatrix(ts->networkOutput.derivative, 0.0f));

        for (auto &cd : ts->connectionData) {
          cd.haveActivation = false;

          // This is so we dont zero out the bias column.
          CuMatrix trimmed = cd.activation;
          trimmed.cols--;
          executor.Execute(Task::FillMatrix(trimmed, 0.0f));
          // executor.Execute(Task::FillMatrix(cd.derivative, 0.0f));
        }
      }
    }
  }

  void workerForwardprop(TaskExecutor &executor, unsigned workerIdx) {
    // Forward Prop has no 'trivial' stream level parallelism.
    if (workerIdx != 0) {
      return;
    }

    for (int i = 0; i < static_cast<int>(curTraceLength); i++) {
      // Copy the input/output from host to device memory.
      traceTargets[i].batchSize = curBatchSize;
      executor.Execute(Task::CopyMatrixH2D(inputOutputStaging[i].second, traceTargets[i].value));

      CuTimeSlice *ts = layerMemory.GetTimeSlice(i);
      assert(ts != nullptr);

      bool foundInput = false;
      for (auto &cd : ts->connectionData) {
        if (cd.connection.srcLayerId == 0) {
          executor.Execute(Task::CopyMatrixH2D(inputOutputStaging[i].first, cd.activation));
          cd.haveActivation = true;
          foundInput = true;
        }
      }
      assert(foundInput);

      // Do the forward pass through this time slice.
      forwardProp(executor, i);
      assert(ts->networkOutput.haveActivation);
    }
  }

  void workerBackpropDelta(TaskExecutor &executor, unsigned workerIdx) {
    // BackProp of deltas has no 'trivial' stream level parallelism.
    if (workerIdx != 0) {
      return;
    }

    for (int i = static_cast<int>(curTraceLength) - 1; i >= 0; i--) {
      backProp(executor, i);
    }
  }

  void workerComputeAndUpdateGradients(TaskExecutor &executor, unsigned workerIdx) {
    unsigned skip = workers.size();

    for (unsigned i = workerIdx; i < spec.connections.size(); i += skip) {
      LayerConnection connection = spec.connections[i];
      CuConnectionAccum *connAccum = gradientAccum.GetConnection(connection);
      assert(connAccum != nullptr);

      for (int timestamp = 0; timestamp < static_cast<int>(curTraceLength); timestamp++) {
        if (connection.timeOffset == 1 && timestamp == 0) {
          continue;
        }

        CuTimeSlice *ts = layerMemory.GetTimeSlice(timestamp);
        assert(ts != nullptr);

        CuConnectionMemoryData *connData = ts->GetConnectionData(connection);
        assert(connData != nullptr && connData->haveActivation);

        CuLayerAccum *layerDelta = deltaAccum.GetDelta(connection.dstLayerId, timestamp);
        assert(layerDelta != nullptr);

        ConnectionActivation activationIn(curBatchSize, connData->activation, connData->derivative);

        executor.Execute(
            Task::GradientIncrement(LayerBatchDeltas(curBatchSize, layerDelta->accumDelta),
                                    activationIn, connAccum->accumGradient));
        connAccum->samples++;
      }

      float scaleFactor = 1.0f / static_cast<float>(curBatchSize * connAccum->samples);
      executor.Execute(Task::ScaleMatrix(connAccum->accumGradient, scaleFactor));

      CuAdamConnection *adamConn = adamState.GetConnection(connection);
      assert(adamConn != nullptr);

      executor.Execute(Task::AdamUpdate(connAccum->accumGradient, adamConn->momentum, adamConn->rms,
                                        ADAM_BETA1, ADAM_BETA2));

      CuLayer *dstLayer = findLayer(connection.dstLayerId);
      assert(dstLayer != nullptr);

      CuWeights *weights = dstLayer->GetWeights(connection);
      assert(weights != nullptr);

      executor.Execute(Task::AdamIncrement(weights->weights, adamConn->momentum, adamConn->rms,
                                           ADAM_BETA1, ADAM_BETA2, ADAM_LR, ADAM_EPSILON));
      executor.Execute(Task::TransposeMatrix(weights->weights, weights->weightsT));
    }
  }

  void forwardProp(TaskExecutor &executor, int timestamp) {
    for (auto &layer : layers) {
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

        ConnectionActivation activationIn(curBatchSize, inData->activation, inData->derivative);
        executor.Execute(
            Task::ForwardIncrement(in.second.weights, activationIn, targetOut->activation));
      }

      ConnectionActivation outActivation(curBatchSize, targetOut->activation,
                                         targetOut->derivative);
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

    if (layer.isOutput) {
      CuTimeSlice *ts = layerMemory.GetTimeSlice(timestamp);
      assert(ts != nullptr);
      result.push_back(&ts->networkOutput);
    } else {
      for (auto &conn : layer.outgoing) {
        CuConnectionMemoryData *cmd = getConnectionMemoryData(conn, timestamp + conn.timeOffset);
        if (cmd != nullptr) {
          result.push_back(cmd);
        }
      }
    }

    return result;
  }

  void backProp(TaskExecutor &executor, int timestamp) {
    CuTimeSlice *ts = layerMemory.GetTimeSlice(timestamp);
    assert(ts != nullptr);

    ConnectionActivation networkOut(curBatchSize, ts->networkOutput.activation,
                                    ts->networkOutput.derivative);

    CuLayerAccum *outputDelta = deltaAccum.GetDelta(layers.back().layerId, timestamp);
    assert(outputDelta != nullptr && outputDelta->samples == 0);

    executor.Execute(Task::ErrorMeasure(networkOut, traceTargets[timestamp],
                                        LayerBatchDeltas(curBatchSize, outputDelta->accumDelta)));
    outputDelta->samples = 1;

    assert(layers.back().isOutput);
    recursiveBackprop(executor, layers.back(), timestamp);
  }

  void recursiveBackprop(TaskExecutor &executor, const CuLayer &layer, int timestamp) {
    CuLayerAccum *layerDelta = deltaAccum.GetDelta(layer.layerId, timestamp);
    assert(layerDelta != nullptr);

    assert(layerDelta->samples > 0);
    if (layerDelta->samples > 1) {
      float deltaScale = 1.0f / static_cast<float>(layerDelta->samples);
      executor.Execute(Task::ScaleMatrix(layerDelta->accumDelta, deltaScale));
    }

    LayerBatchDeltas batchDelta(curBatchSize, layerDelta->accumDelta);

    CuTimeSlice *slice = layerMemory.GetTimeSlice(timestamp);
    assert(slice != nullptr);

    for (const auto &connection : layer.incoming) {
      if (connection.first.timeOffset == 1 && timestamp == 0) {
        continue;
      }

      if (connection.first.srcLayerId != 0) { // The source is another layer from the srcSlice.
        CuConnectionMemoryData *connData = slice->GetConnectionData(connection.first);
        assert(connData != nullptr && connData->haveActivation);

        ConnectionActivation activationIn(curBatchSize, connData->activation, connData->derivative);

        int srcTimestamp = timestamp - connection.first.timeOffset;
        assert(srcTimestamp >= 0);

        CuLayer *srcLayer = findLayer(connection.first.srcLayerId);
        assert(srcLayer != nullptr);

        CuLayerAccum *srcDelta = deltaAccum.GetDelta(connection.first.srcLayerId, srcTimestamp);
        assert(srcDelta != nullptr);

        LayerBatchDeltas targetDelta(curBatchSize, srcDelta->accumDelta);
        executor.Execute(Task::PropagateDelta(batchDelta, connection.second.weightsT, activationIn,
                                              targetDelta));
        srcDelta->samples++;

        if (connection.first.timeOffset == 0) {
          recursiveBackprop(executor, *srcLayer, timestamp);
        }
      }
    }
  }

  CuConnectionMemoryData *getConnectionMemoryData(const LayerConnection &conn, int timestamp) {
    CuTimeSlice *ts = layerMemory.GetTimeSlice(timestamp);
    if (ts == nullptr) {
      return nullptr;
    }

    auto r = ts->GetConnectionData(conn);
    assert(r != nullptr);
    return r;
  }

  CuLayer *findLayer(unsigned layerId) {
    for (auto &layer : layers) {
      if (layer.layerId == layerId) {
        return &layer;
      }
    }
    return nullptr;
  }
};

CudaTrainer::CudaTrainer(const RNNSpec &spec) : impl(new CudaTrainerImpl(spec)) {}

CudaTrainer::~CudaTrainer() = default;

void CudaTrainer::SetWeights(const vector<pair<LayerConnection, math::MatrixView>> &weights) {
  impl->SetWeights(weights);
}

void CudaTrainer::GetWeights(vector<pair<LayerConnection, math::MatrixView>> &outWeights) {
  impl->GetWeights(outWeights);
}

void CudaTrainer::Train(const vector<SliceBatch> &trace) { impl->Train(trace); }
