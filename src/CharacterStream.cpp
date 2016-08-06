
#include "CharacterStream.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>

struct CharacterStream::CharacterStreamImpl {
  vector<char> mappedChars;

  std::ifstream fileStream;
  int prevChar;

  CharacterStreamImpl(const string &filePath) : fileStream(filePath), prevChar(0) {
    initMappedChars();
  }

  unsigned VectorDimension(void) const { return mappedChars.size(); }

  Maybe<math::OneHotVector> ReadCharacter(void) {
    while (true) {
      int nextChar = fileStream.get();
      if (nextChar == EOF) {
        return Maybe<math::OneHotVector>::none;
      }

      nextChar = normalisedCharacter(nextChar);
      if (nextChar == ' ' && nextChar == prevChar) {
        continue;
      }

      auto mappedIter = find(mappedChars.begin(), mappedChars.end(), nextChar);
      if (mappedIter == mappedChars.end()) {
        continue;
      }

      prevChar = nextChar;

      unsigned index = mappedIter - mappedChars.begin();
      assert(index < mappedChars.size());

      return Maybe<math::OneHotVector>(math::OneHotVector(mappedChars.size(), index));
    }

    return Maybe<math::OneHotVector>::none;
  }

  vector<math::OneHotVector> ReadCharacters(unsigned max) {
    vector<math::OneHotVector> result;
    result.reserve(max);

    for (unsigned i = 0; i < max; i++) {
      auto next = ReadCharacter();
      if (next.valid()) {
        result.push_back(next.val());
      } else {
        break;
      }
    }

    return result;
  }

  char Decode(unsigned index) const {
    assert(index < mappedChars.size());
    return mappedChars[index];
  }

  void initMappedChars(void) {
    mappedChars = {' ', '.', '!', '?', '\"', '\'', '(', ')', '[', ']', '{', '}', '-',
                   '@', '#', '$', '%', '&',  '*',  '<', '>', ':', ';', '/', '\\'};

    for (char c = 'a'; c <= 'z'; c++) {
      mappedChars.push_back(c);
    }
    for (char c = '0'; c <= '9'; c++) {
      mappedChars.push_back(c);
    }
  }

  int normalisedCharacter(int curChar) {
    if (isspace(curChar)) {
      return ' ';
    } else {
      return tolower(curChar);
    }
  }
};

CharacterStream::CharacterStream(const string &filePath)
    : impl(new CharacterStreamImpl(filePath)) {}

CharacterStream::~CharacterStream() = default;

unsigned CharacterStream::VectorDimension(void) const { return impl->VectorDimension(); }

Maybe<math::OneHotVector> CharacterStream::ReadCharacter(void) { return impl->ReadCharacter(); }

vector<math::OneHotVector> CharacterStream::ReadCharacters(unsigned max) {
  return impl->ReadCharacters(max);
}

char CharacterStream::Decode(unsigned index) const { return impl->Decode(index); }
