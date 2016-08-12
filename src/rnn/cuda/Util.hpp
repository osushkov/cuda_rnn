#pragma once

#include "Types.hpp"
#include <cstdlib>

namespace rnn {
namespace cuda {
namespace util {

void *AllocPinned(size_t bufSize);
void FreePinned(void *buf);

CuMatrix AllocMatrix(unsigned rows, unsigned cols);
void FreeMatrix(CuMatrix &m);
}
}
}
