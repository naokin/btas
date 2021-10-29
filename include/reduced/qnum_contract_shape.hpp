

template<size_t N, class Q>
Tensor<double,2ul> get_CG_matrix (const SpShape<Q>& q1, const SpShape<Q>& q2)
{
  Tensor<double,2ul> CG_matrix(q1.size(),q2.size());
  CG_matrix = 0.0;

}
