#pragma once

#include "common/Common.hpp"
#include "common/Maybe.hpp"
#include "math/OneHotVector.hpp"

#include <vector>

class CharacterStream {
public:
  CharacterStream(const string &filePath);
  ~CharacterStream();

  unsigned VectorDimension(void) const;

  Maybe<math::OneHotVector> ReadCharacter(void);
  vector<math::OneHotVector> ReadCharacters(unsigned max);

  char Decode(unsigned index) const;

private:
  struct CharacterStreamImpl;
  uptr<CharacterStreamImpl> impl;
};
