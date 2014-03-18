#ifndef __BTAS_COMMON_LEADING_RANK_H
#define __BTAS_COMMON_LEADING_RANK_H 1

#include <btas/common/types.h>

namespace btas
{

/// traits to return leading rank of tensor, which is defined as antonym of stride_rank
template<size_t N, CBLAS_ORDER Order> struct leading_rank { };

template<size_t N>
struct leading_rank<N, CblasRowMajor> { static constexpr const size_t value = 0; };

template<size_t N>
struct leading_rank<N, CblasColMajor> { static constexpr const size_t value = N-1; };

} // namespace btas

#endif // __BTAS_COMMON_LEADING_RANK_H
