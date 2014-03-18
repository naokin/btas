#ifndef __BTAS_COMMON_ASSERT_H
#define __BTAS_COMMON_ASSERT_H 1

#include <cassert>
#include <stdexcept>

#define BTAS_ASSERT(truth, msg) { if (!(truth)) { throw std::runtime_error(msg); } }

#define BTAS_STATIC_ASSERT static_assert

#include <iostream>

#define BTAS_WARNING(msg) { std::cout << "WARNING: " << msg << std::endl; }

#define BTAS_DEBUG(msg) { std::cout << "DEBUG: " << msg << std::endl; }

#endif // __BTAS_COMMON_ASSERT_H
