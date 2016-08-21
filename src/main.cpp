
#include "CharacterStream.hpp"
#include "RNNSampler.hpp"
#include "RNNTrainer.hpp"
#include "common/Common.hpp"
#include <string>

void testRNN(string path) {
  CharacterStream cstream(path);

  RNNTrainer trainer(16);
  auto network = trainer.TrainLanguageNetwork(cstream, 1000);

  RNNSampler sampler(cstream.VectorDimension());
  vector<unsigned> sampled = sampler.SampleCharacters(network.get(), 1000);

  for (const auto sample : sampled) {
    cout << cstream.Decode(sample);
  }

  cout << endl;
}

int main(int argc, char **argv) {
  srand(1234);

  cout << "testing rnn" << endl;
  testRNN(string(argv[1]));

  return 0;
}
