
// reshape concept
{
  // A = {a_i,j,k,l}
  Tensor<double,4ul> A(2,2,2,2);
  // B = {a_ij,kl}
  auto B = A.reshape(4,4);
  // C = {a_ik,jl}
  auto C = A.permute(0,2,1,3).reshape(4,4);
}

