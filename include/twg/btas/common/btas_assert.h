#ifndef __BTAS_ASSERT_H
#define __BTAS_ASSERT_H 1

#include <cassert>
#include <stdexcept>

#define BTAS_ASSERT(truth, msg) { if (!(truth)) { throw std::runtime_error(msg); } }

#define BTAS_STATIC_ASSERT static_assert

#endif // __BTAS_ASSERT_H
