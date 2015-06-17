#ifndef __BTAS_REMOVE_COMPLEX_H
#define __BTAS_REMOVE_COMPLEX_H

#include <complex>

namespace btas {

template<typename T> struct remove_complex { typedef T type; };

template<typename T> struct remove_complex<std::complex<T> > { typedef T type; };

} // namespace btas

#endif // __BTAS_REMOVE_COMPLEX_H
