
// design how to merge quantum sparse tensor?
{
  using namespace btas;

  BlockSpTensor<double,4,fermion> psi2;

  BlockSpMerge<double,2,2,fermion> psim = make_merge<2,2>(psi2);

  BlockSpMerge<double,2,NO_MERGE,fermion> psir = make_merge<2,NO_MERGE>(psi2);

  BlockSpMerge<double,NO_MERGE,2,fermion> psic = make_merge<NO_MERGE,2>(psi2);
}


template<typename T, size_t R, size_t C, class Q, CBLAS_ORDER Order = CblasRowMajor>
class BlockSpMerge : public BlockSpTensor<T,2,Q,Order>
{

private:

  MergeInfo<R,Q> rowinfo_;

  MergeInfo<C,Q> colinfo_;

};

// TensorWrapper
{
  Tensor<double,4> A(4,4,4,4);
  // reshaped reference of A having continuous storage concept
  TensorWrapper<double*,2> B(A.data(),shape(16,16));

  std::vector<double> V(100);
  // 1d vector can be viewed as tensor
  TensorWrapper<double*,3> C(V.data(),shape(4,5,5));
  // wrap data w/ const-qualifier
  TensorWrapper<const double*,3> D(C);
}

// TensorView
{
  Tensor<double,4> A(4,4,4,4);
  // this is same as TensorView to get the reshaped reference of A
  TensorView<double*,2> B(A.data(),shape(16,16));
  // TensorView can take any random-access iterator (including TensorIterator)
  // to provide a multi-layered reference of a tensor.
  TensorView<TensorView<double*,2>::iterator,2> C(B.begin(),B.extent(),B.stride());

  // which
  TensorView<TensorView<const double*,2>::iterator,2> D(C);
  // or
  TensorView<TensorView<double*,2>::const_iterator,2> D(C);
  // is better?

  TensorIterator<TensorIterator<Tensor<double,4>::iterator,4>,4> i;

  TensorIterator<TensorIterator<Tensor<double,4>::const_iterator,4>,4> j;
}
