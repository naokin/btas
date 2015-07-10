Sparse and part of SpTensorBase to be associated as SpShape


template<size_t N, class Q>
class SpShape {
public:
  //
  typedef Q qnum_type;
  //
  typedef std::vector<qnum_type> qnum_array_type;
  //
  typedef std::array<qnum_array_type> qnum_shape_type;
  //
  SpShape (const qnum_type& q, const qnum_shape_type& qs);
  //
  SpShape (const SpShape& x);
  // construct shape of 'x * y'
  template<size_t L, size_t M>
  SpShape (const SpShape<L,Q>& x, const SpShape<M,Q>& y);
  //
  SpShape& operator= (const SpShape& x);
  //
  template<class Index_>
  bool is_allowed (const Index_& idx) const;
  //
  const qnum_type& qnum () const;
  //
  const qnum_shape_type& qnum_shape () const;
  //
  const qnum_array_type& qnum_array (size_t i) const;
  //
  void clear ();
  //
  void swap (SpShape& x);
private:
  //
  qnum_type qnum_;
  //
  qnum_shape_type qnum_shape_;
};

template<typename T, size_t N, class Q, CBLAS_ORDER Order = CblasRowMajor>
class SpTensor : public SpShape<N,Q,Order> {

public:

  size_t nnz_local () const;

  size_t nnz () const;

};

template<typename T, size_t N, class Q, CBLAS_ORDER Order = CblasRowMajor>
class BlockSpTensor : public SpTensor<Tensor<T,N,Order>,N,Q,Order> {
};


