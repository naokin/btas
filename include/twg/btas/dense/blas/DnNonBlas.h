#ifndef __BTAS_DENSE_NON_BLAS_H
#define __BTAS_DENSE_NON_BLAS_H 1

namespace btas
{

/// copy with reshape
template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
void DnCopyR (const DnTensor<T, M, Order>& X, DnTensor<T, N, Order>& Y)
{
   BTAS_ASSERT(X.size() == Y.size(), "DnCopyR: mismatched size for Y");
   copy(X.size(), X.data(), 1, Y.data(), 1);
}

/// axpy with reshape
template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
void DnAXpYR (const T& alpha, const DnTensor<T, M, Order>& X, DnTensor<T, N, Order>& Y)
{
   BTAS_ASSERT(X.size() == Y.size(), "DnAXpYR: mismatched size for Y");
   axpy(X.size(), alpha, X.data(), 1, Y.data(), 1);
}

template<typename T>
void DnDiMV_impl (const T& alpha, const T* A, const T* X, const size_t& incX, const T& beta, T* Y, const size_t& incY)
{
   if(beta == NumType<T>::one())
   {
      for(size_t i = 0; i < A.size(); ++i)
      {
         (*Y) += alpha*(*A)*(*X);
         ++A;
         X += incX;
         Y += incY;
      }
   }
   else
   {
      for(size_t i = 0; i < A.size(); ++i)
      {
         (*Y) *= beta;
         (*Y) += alpha*(*A)*(*X);
         ++A;
         X += incX;
         Y += incY;
      }
   }
}

template<typename T, size_t N, CBLAS_ORDER Order>
void DnDiMV (
   const T& alpha,
   const DnTensor<T, N, Order>& A,
   const DnTensor<T, N, Order>& X,
   const T& beta,
         DnTensor<T, N, Order>& Y)
{
   BTAS_ASSERT(A.extent() == X.extent(), "DnDiMV: mismatched shape for X");
   if(Y.size() > 0)
   {
      BTAS_ASSERT(A.extent() == Y.extent(), "DnDiMV: mismatched shape for Y");
   }
   else
   {
      Y.resize(X.extent(), NumType<T>::zero());
   }
   DnDiMV_impl(alpha, A.data(), X.data(), 1, beta, Y.data(), 1);
}

template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
void DnDiMM (
   const CBLAS_TRANSPOSE& transB,
   const T& alpha,
   const DnTensor<T, M, Order>& A, // diagonal tensor e.g. A(i,j,i,j)
   const DnTensor<T, N, Order>& B, // general tensor e.g. B(i,j,...)
   const T& beta,
         typename std::enable_if<(M < N), DnTensor<T, N, Order>>::type& C)
{
   dimm_contract_shape<M, N> cs(Order, transB, A.extent(), B.extent());
   if(C.size() > 0)
   {
      BTAS_ASSERT(C.extent() == cs.shapeC, "DnDiMM: mismatched shape for C");
      if(beta != NumType<T>::one()) scal(C.size(), beta, C.data(), 1);
   }
   else
   {
      C.resize(cs.shapeC, NumType<T>::zero());
   }

   const T* ptrA = A.data();
   const T* ptrB = B.data();
         T* ptrC = C.data();

   if(transB == CblasNoTrans)
   {
      if(Order == CblasRowMajor)
      {
         for(size_t i = 0; i < cs.rowsB; ++i)
         {
            axpy(cs.ldB, alpha*(*ptrA), ptrB, 1, ptrC, 1);
            ++ptrA;
            ptrB += cs.ldB;
            ptrC += cs.ldC;
         }
      }
      else
      {
         for(size_t i = 0; i < cs.colsB; ++i)
         {
            DnDiMV_impl(alpha, ptrA, ptrB, 1, NumType<T>::one(), ptrC, 1);
            ptrB += cs.ldB;
            ptrC += cs.ldC;
         }
      }
   }
   else
   {
      if(Order == CblasRowMajor)
      {
         for(size_t i = 0; i < cs.colsB; ++i)
         {
            DnDiMV_impl(alpha, ptrA, ptrB, 1, NumType<T>::one(), ptrC, cs.ldC);
            ptrB += cs.ldB;
            ++ptrC;
         }
      }
      else
      {
         for(size_t i = 0; i < cs.rowsB; ++i)
         {
            axpy(cs.ldB, alpha*(*ptrA), ptrB, 1, ptrC, cs.ldC);
            ++ptrA;
            ptrB += cs.ldB;
            ++ptrC;
         }
      }
   }
}

template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
void DnDiMM (
   const CBLAS_TRANSPOSE& transA,
   const T& alpha,
   const DnTensor<T, M, Order>& A, // general tensor e.g. A(...,i,j)
   const DnTensor<T, N, Order>& B, // diagonal tensor e.g. B(i,j,i,j)
   const T& beta,
         typename std::enable_if<(M > N), DnTensor<T, M, Order>>::type& C)
{
   dimm_contract_shape<M, N> cs(Order, transA, A.extent(), B.extent());
   if(C.size() > 0)
   {
      BTAS_ASSERT(C.extent() == cs.shapeC, "DnDiMM: mismatched shape for C");
      if(beta != NumType<T>::one()) scal(C.size(), beta, C.data(), 1);
   }
   else
   {
      C.resize(cs.shapeC, NumType<T>::zero());
   }

   const T* ptrA = A.data();
   const T* ptrB = B.data();
         T* ptrC = C.data();

   if(transA == CblasNoTrans)
   {
      if(Order == CblasRowMajor)
      {
         for(size_t i = 0; i < cs.rowsA; ++i)
         {
            DnDiMV_impl(alpha, ptrA, ptrB, 1, NumType<T>::one(), ptrC, 1);
            ptrA += cs.ldA;
            ptrC += cs.ldC;
         }
      }
      else
      {
         for(size_t i = 0; i < cs.colsA; ++i)
         {
            axpy(cs.ldA, alpha*(*ptrB), ptrA, 1, ptrC, 1);
            ++ptrB;
            ptrA += cs.ldA;
            ptrC += cs.ldC;
         }
      }
   }
   else
   {
      if(Order == CblasRowMajor)
      {
         for(size_t i = 0; i < cs.colsA; ++i)
         {
            axpy(cs.ldA, alpha*(*ptrB), ptrA, 1, ptrC, cs.ldC);
            ++ptrB;
            ptrA += cs.ldA;
            ++ptrC;
         }
      }
      else
      {
         for(size_t i = 0; i < cs.rowsA; ++i)
         {
            DnDiMV_impl(alpha, ptrA, ptrB, 1, NumType<T>::one(), ptrC, cs.ldC);
            ptrA += cs.ldA;
            ++ptrC;
         }
      }
   }
}

} // namespace btas

#endif // __BTAS_DENSE_NON_BLAS_H
