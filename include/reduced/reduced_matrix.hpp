#ifndef __BTAS_REDUCED_MATRIX_HPP__
#define __BTAS_REDUCED_MATRIX_HPP__

namespace btas {

template<typename T, size_t N, class Q, CBLAS_ORDER Order = CblasRowMajor>
class reduced_matrix
{



private:

  qnum_shape_type iq_; ///< incoming quantum #s

  qnum_shape_type oq_; ///< outgoing quantum #s

  SpTensor<T,2ul,Q,Order> data_;

}; // class reduced_matrix

} // namespace btas

#endif // __BTAS_REDUCED_MATRIX_HPP__
