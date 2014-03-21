#ifndef __BTAS_COMMON_ASSERT_H
#define __BTAS_COMMON_ASSERT_H 1

#include <cassert>
#include <stdexcept>

#define BTAS_ASSERT(truth, msg) { if (!(truth)) { std::cout << msg << std::endl; assert(truth); } }

#define BTAS_STATIC_ASSERT static_assert

#include <iostream>

#ifdef _ENABLE_BTAS_WARNING
#define BTAS_WARNING(msg) { std::cout << "WARNING: " << msg << std::endl; }
#else
#define BTAS_WARNING(msg) { }
#endif

#ifdef _ENABLE_BTAS_DEBUG
#define BTAS_DEBUG(msg) { std::cout << "DEBUG: " << msg << std::endl; }
#else
#define BTAS_DEBUG(msg) { }
#endif

#endif // __BTAS_COMMON_ASSERT_H
