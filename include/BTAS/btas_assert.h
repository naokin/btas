#ifndef __BTAS_ASSERT_H
#define __BTAS_ASSERT_H

#include <stdexcept>

namespace btas {

#define BTAS_ASSERT(flag, msg) { if(!(flag)) { throw std::runtime_error(msg); } }

} // namespace btas

#endif // __BTAS_ASSERT_H
