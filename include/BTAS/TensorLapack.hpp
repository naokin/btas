#ifndef __BTAS_TENSOR_LAPACK_HPP
#define __BTAS_TENSOR_LAPACK_HPP

#include <vector>
#include <algorithm>

#include <BTAS/btas_assert.h>
#include <BTAS/remove_complex.h>

#include <LAPACK/wrappers.h>

#ifndef __BTAS_TENSOR_HPP
#include <BTAS/Tensor.hpp>
#endif

namespace btas {

/// Solve real-symmetric eigenvalue problem (SEP)
/// Def.: A({i,j,k},{i,j,k}) = Z({i,j,k,e}) * w({e}) * Z^T({e,i,j,k})
/// NOTE: if called with complex array, gives an error
template<typename T, size_t N, CBLAS_ORDER Order>
void syev (
  const char& jobz,
  const char& uplo,
  const Tensor<T,2*N-2,Order>& a,
        Tensor<T,1,Order>& w,
        Tensor<T,N,Order>& z)
{
  const size_t K = N-1;

  BTAS_ASSERT(std::equal(a.extent().begin(),a.extent().begin()+K,a.extent().begin()+K),"input tensor is not symmetric.");

  size_t aCols = std::accumulate(a.extent().begin()+K,a.extent().end(),1ul,std::multiplies<size_t>());

  typename Tensor<T,N,Order>::extent_type zExtent;
  for(size_t i = 0; i < K; ++i) zExtent[i] = a.extent(i);
  zExtent[N-1] = aCols;

  z.resize(zExtent);
  copy(a,z); // reshape a and copy to z

  w.resize(aCols);

  syev(Order,jobz,uplo,aCols,z.data(),aCols,w.data());
}

/// Solve hermitian eigenvalue problem (HEP)
/// Def.: A({i,j,k},{i,j,k}) = Z({i,j,k,e}) * w({e}) * Z^T({e,i,j,k})
/// NOTE: if called with real array, redirect to Syev
template<typename T, size_t N, CBLAS_ORDER Order>
void heev (
  const char& jobz,
  const char& uplo,
  const Tensor<T,2*N-2,Order>& a,
        Tensor<typename remove_complex<T>::type,1,Order>& w,
        Tensor<T,N,Order>& z)
{
  const size_t K = N-1;

  BTAS_ASSERT(std::equal(a.extent().begin(),a.extent().begin()+K,a.extent().begin()+K),"input tensor is not symmetric.");

  size_t aCols = std::accumulate(a.extent().begin()+K,a.extent().end(),1ul,std::multiplies<size_t>());

  typename Tensor<T,N,Order>::extent_type zExtent;
  for(size_t i = 0; i < K; ++i) zExtent[i] = a.extent(i);
  zExtent[N-1] = aCols;

  z.resize(zExtent); copy(a,z); // reshape a and copy to z

  w.resize(aCols);

  heev(Order,jobz,uplo,aCols,z.data(),aCols,w.data());
}

/// Solve singular value decomposition (SVD)
template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
void gesvd (
  const char& jobu,
  const char& jobvt,
  const Tensor<T,M+N-2,Order>& a,
        Tensor<typename remove_complex<T>::type,1,Order>& s,
        Tensor<T,M,Order>& u,
        Tensor<T,N,Order>& vt)
{
  size_t aRows = std::accumulate(a.extent().begin(),a.extent().begin()+M-1,1ul,std::multiplies<size_t>());
  size_t aCols = std::accumulate(a.extent().begin()+M-1,a.extent().end(),  1ul,std::multiplies<size_t>());
  size_t lda = (Order == CblasRowMajor) ? aCols : aRows;

  size_t sExts = std::min(aRows,aCols);

  size_t uCols = (jobu == 'A' || jobu == 'a') ? aRows : sExts;
  size_t ldu = (Order == CblasRowMajor) ? aRows : uCols;

  size_t vtRows = (jobvt == 'A' || jobvt == 'a') ? aCols : sExts;
  size_t ldvt = (Order == CblasRowMajor) ? aCols : vtRows;

  typename Tensor<T,M,Order>::extent_type uExtent;
  for(size_t i = 0; i < M-1; ++i) uExtent[i] = a.extent(i);
  uExtent[M-1] = uCols;

  typename Tensor<T,N,Order>::extent_type vtExtent;
  vtExtent[0] = vtRows;
  for(size_t i = 1; i < N; ++i) vtExtent[i] = a.extent(i+M-2);

  s.resize(sExts);

  BTAS_ASSERT((jobu == 'O' || jobu == 'o' || jobvt == 'O' || jobvt == 'o'), "job* = 'O' is currently disabled.")

  if(jobu  != 'N' && jobu  != 'n') u.resize(uExtent);
  if(jobvt != 'N' && jobvt != 'n') vt.resize(vtExtent);

  Tensor<T,M+N-2,Order> aCp(a);
  gesvd(Order,jobu,jobvt,aRows,aCols,aCp.data(),lda,s.data(),u.data(),ldu,vt.data(),ldvt);
}

/// perform a QR decomposition : a = q * r
/// \param a input tensor
/// \param q on exit, unitary matrix is stored
/// \param r on exit, upper trapezoidal matrix is stored
template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
void geqrf (
  const Tensor<T,M+N-2,Order>& a,
        Tensor<T,M,Order>& q,
        Tensor<T,N,Order>& r)
{
  size_t aRows = std::accumulate(a.extent().begin(),a.extent().begin()+M-1,1ul,std::multiplies<size_t>());
  size_t aCols = std::accumulate(a.extent().begin()+M-1,a.extent().end(),  1ul,std::multiplies<size_t>());
  size_t lda = (Order == CblasRowMajor) ? aCols : aRows;

  size_t tExts = std::min(aRows,aCols);

  typename Tensor<T,M,Order>::extent_type qExtent;
  for(size_t i = 0; i < M-1; ++i) qExtent[i] = a.extent(i);
  qExtent[M-1] = tExts;

  typename Tensor<T,N,Order>::extent_type rExtent;
  rExtent[0] = tExts;
  for(size_t i = 1; i < N; ++i) rExtent[i] = a.extent(i+M-2);

  Tensor<T,2,Order> aCp(aRows,aCols);
  copy(a,aCp);

  std::vector<double> tau(tExts);
  geqrf(Order,aRows,aCols,aCp.data(),lda,tau.data());

  r.resize(rExtent);
  r.fill(static_cast<T>(0));

  q.resize(qExtent);

  // NOTE: by def. tExts <= aCols
  if(Order == CblasRowMajor) {
    for(size_t i = 0; i < tExts; ++i)
      for(size_t j = i; j < aCols; ++j)
        r[i*aCols+j] = aCp(i,j);

    for(size_t i = 0; i < aRows; ++i)
      for(size_t j = 0; j < tExts; ++j)
        q[i*tExts+j] = aCp(i,j);
  }
  else {
    for(size_t j = 0; j < tExts; ++j)
      for(size_t i = 0; i <= j; ++i)
        r[i+j*tExts] = aCp(i,j);
    for(size_t j = tExts; j < aCols; ++j)
      for(size_t i = 0; i < tExts; ++i)
        r[i+j*tExts] = aCp(i,j);

    for(size_t j = 0; j < tExts; ++j)
      for(size_t i = 0; i < aRows; ++i)
        q[i+j*aRows] = aCp(i,j);
  }

  // Now get the Q matrix out
  size_t ldq = (Order == CblasRowMajor) ? tExts : aRows;
  orgqr(Order,aRows,tExts,tExts,q.data(),ldq,tau.data());
}

/// perform a LQ decomposition : a = l * q
/// \param a input tensor
/// \param l on exit, lower trapezoidal matrix is stored
/// \param q on exit, unitary matrix is stored
template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
void gelqf (
  const Tensor<T,M+N-2,Order>& a,
        Tensor<T,M,Order>& l,
        Tensor<T,N,Order>& q)
{
  size_t aRows = std::accumulate(a.extent().begin(),a.extent().begin()+M-1,1ul,std::multiplies<size_t>());
  size_t aCols = std::accumulate(a.extent().begin()+M-1,a.extent().end(),  1ul,std::multiplies<size_t>());
  size_t lda = (Order == CblasRowMajor) ? aCols : aRows;

  size_t tExts = std::min(aRows,aCols);

  typename Tensor<T,M,Order>::extent_type lExtent;
  for(size_t i = 0; i < M-1; ++i) lExtent[i] = a.extent(i);
  lExtent[M-1] = tExts;

  typename Tensor<T,N,Order>::extent_type qExtent;
  qExtent[0] = tExts;
  for(size_t i = 1; i < N; ++i) qExtent[i] = a.extent(i+M-2);

  Tensor<T,2,Order> aCp(aRows,aCols);
  copy(a,aCp);

  std::vector<double> tau(tExts);
  gelqf(Order,aRows,aCols,aCp.data(),lda,tau.data());

  l.resize(lExtent);
  l.fill(static_cast<T>(0));

  q.resize(qExtent);

  // NOTE: by def. tExts <= aCols
  if(Order == CblasRowMajor) {
    for(size_t i = 0; i < tExts; ++i)
      for(size_t j = 0; j <= i; ++j)
        l[i*tExts+j] = aCp(i,j);
    for(size_t i = tExts; i < aRows; ++i)
      for(size_t j = 0; j < tExts; ++j)
        l[i*tExts+j] = aCp(i,j);

    for(size_t i = 0; i < tExts; ++i)
      for(size_t j = 0; j < aCols; ++j)
        q[i*aCols+j] = aCp(i,j);
  }
  else {
    for(size_t j = 0; j < tExts; ++j)
      for(size_t i = j; i < aRows; ++i)
        l[i+j*aRows] = aCp(i,j);

    for(size_t j = 0; j < aCols; ++j)
      for(size_t i = 0; i < tExts; ++i)
        q[i+j*tExts] = aCp(i,j);
  }

  // Now get the Q matrix out
  size_t ldq = (Order == CblasRowMajor) ? aCols : tExts;
  orglq(Order,tExts,aCols,tExts,q.data(),ldq,tau.data());
}

} // namespace btas

#endif // __BTAS_TENSOR_LAPACK_HPP
