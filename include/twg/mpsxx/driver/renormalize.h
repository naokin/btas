#ifndef __BTAS_MPSXX_DRIVER_RENORMALIZE_H
#define __BTAS_MPSXX_DRIVER_RENORMALIZE_H 1

#include <btas/mpsxx/common/types.h>

namespace btas
{

//  NOTE:
//
//  template<typename T, class Qn, size_t N = 1>
//  using MPS = QnTensor<Qn, SpTensor<DnTensor<T, 2, CblasRowMajor>, N+2, CblasRowMajor>>;
// 
//  template<typename T, class Qn, size_t N = 1>
//  using MPO = QnTensor<Qn, SpTensor<T, 2*N+2, CblasRowMajor>>;
// 
//  template<typename T, class Qn>
//  using BLOCK = QnTensor<Qn, SpTensor<DnTensor<T, 2, CblasRowMajor>, 3, CblasRowMajor>>;
// 

/// renormalize operator: i.e. calculate G = W * A^dag * F * B (forward)
///
/// input tensors:
/// W [1,6,4,5]
/// F [1,2,3]
/// A [4,2,7]
/// B [5,3,8]
///
/// output tensor
/// G [6,7,8]
///
/// intermediate tensors
/// X [1,4,7,3] = F * A^dag
/// Y [6,5,7,3] = W * X
///
///  +---(2)---A---(7)
///  |         |
///  |        (4)
///  |         |
///  F---(1)---W---(6)
///  |         |
///  |        (5)
///  |         |
///  +---(3)---B---(8)
///
template<class Qn, typename T>
void __renormalize_fwd_sweep (
      const MPO<T, Qn, 1>& W,
      const MPS<T, Qn, 1>& A,
      const MPS<T, Qn, 1>& B,
      const BLOCK<T, Qn>& F,
            BLOCK<T, Qn>& G)
{
   // Explicit implementation of tensor contractions in renormalization

   const QnIndex<Qn>& Q1 = F.qn(0);
   const QnIndex<Qn>& Q2 = F.qn(1);
   const QnIndex<Qn>& Q3 = F.qn(2);
         QnIndex<Qn>  Q4 =-A.qn(0);
   const QnIndex<Qn>& Q5 = W.qn(3);
   const QnIndex<Qn>& Q6 = W.qn(1);
         QnIndex<Qn>  Q7 =-A.qn(2);
   const QnIndex<Qn>& Q8 = B.qn(2);

   BTAS_ASSERT(W.qn(0) == -Q1, "renormalize: mismatched quantum number index #1");
   BTAS_ASSERT(A.qn(1) == F.qn(1), "renormalize: mismatched quantum number index #2");
   BTAS_ASSERT(B.qn(1) == -Q3, "renormalize: mismatched quantum number index #3");
   BTAS_ASSERT(W.qn(2) == A.qn(0), "renormalize: mismatched quantum number index #4");
   BTAS_ASSERT(B.qn(0) == -Q5, "renormalize: mismatched quantum number index #5");

   size_t D1 = Q1.size();
   size_t D2 = Q2.size();
   size_t D3 = Q3.size();
   size_t D4 = Q4.size();
   size_t D5 = Q5.size();
   size_t D6 = Q6.size();
   size_t D7 = Q7.size();
   size_t D8 = Q8.size();

   size_t D23 = D2*D3;
   size_t D27 = D2*D7;
   size_t D65 = D6*D5;
   size_t D73 = D7*D3;

   size_t D473 = D4*D73;

   // X[1,4,7,3] = F[1,2,3] * A[4,2,7]^dag

   QnTensor<Qn, SpTensor<DnTensor<T, 2>, 4, CblasRowMajor>> Y(F.q()-A.q(),Q1,Q4,Q7,Q3);

   auto F_itr = F.begin();
   auto X_itr = X.begin();

   for(size_t i1 = 0; i1 < D1; ++i1)
   {
      size_t F1 = i1*D23;
      size_t X1 = i1*D473;

           F_itr = F.lower_bound(F_itr, F1);
      auto F_end = F.lower_bound(F_itr, F1+D23);

      if(F_itr == F_end) continue;

      auto A_itr = A.begin();

      for(size_t i4 = 0; i4 < D4; ++i4)
      {
         size_t A4 = i4*D27;
         size_t X14 = X1+i4*D73;

              A_itr = A.lower_bound(A_itr, A4);
         auto A_end = A.lower_bound(A_itr, A4+D27);

         if(A_itr == A_end) continue;

         for(; A_itr != A_end; ++A_itr)
         {
            size_t i27 = A_itr->first%D27;
            size_t i2 = i27/D7;
            size_t i7 = i27%D7;

            size_t F12 = F1+i2*D3;
            size_t X147 = X14+i7*D3;

                 F_itr = F.lower_bound(F_itr, F12);
            auto F_jtr = F.lower_bound(F_itr, F12+D3);

            if(F_itr == F_end) continue;

            for(; F_itr != F_jtr; ++F_itr)
            {
               size_t i3 = F_itr->first%D3;

               X_itr = X.get(X_itr, X147+i3);

               gemm(CblasConjTrans, CblasNoTrans, 1.0, A_itr->second, F_itr->second, 1.0, X_itr->second);
            }
         }
      }
   }

   // Y[6,5,7,3] = W[1,6,4,5] * X[1,4,7,3]

   QnTensor<Qn, SpTensor<DnTensor<T, 2>, 4, CblasRowMajor>> Y(W.q()+X.q(),Q6,Q5,Q7,Q3);

   for(auto W_itr = W.begin(); W_itr != W.end(); ++W_itr)
   {
      size_t i14 = W_itr->first/D65;
      size_t i65 = W_itr->first%D65;

      size_t X14 = i14*D73;

      auto X_itr = X.lower_bound(X14);
      auto X_end = X.lower_bound(X_itr, X14+D73);

      if(X_itr == X_end) continue;

      size_t Y65 = i65*D73;

      auto Y_itr = Y.lower_bound(Y65);

      for(; X_itr != X_end; ++X_itr)
      {
         size_t i73 = X_itr->first%D73;

         Y_itr = Y.get(Y_itr, Y65+i73);

         axpy(W_itr->second, X_itr->second, Y_itr->second);
      }
   }

   // G[6,7,8] = Y[6,5,7,3] * B[5,3,8]

   if(G.size() > 0)
   {
      BTAS_ASSERT(G.qn(0) == Q6, "renormalize: mismatched quantum number index #6");
      BTAS_ASSERT(G.qn(1) == Q7, "renormalize: mismatched quantum number index #7");
      BTAS_ASSERT(G.qn(2) == Q8, "renormalize: mismatched quantum number index #8");
   }
   else
   {
      G.resize(Y.q()+B.q(),Q6,Q7,Q8);
   }

   for(auto Y_itr = Y.begin(); Y_itr != Y.end(); ++Y_itr)
   {
      size_t i3 = Y_itr->first%D3;
      size_t i657 = Y_itr->first/D3;
      size_t i7 = i657%D7;
      size_t i65 = i657/D7;
      size_t i5 = i65%D5;
      size_t i6 = i65/D5;

      size_t B53 = i5*D3+i3;
      size_t G67 = i6*D7+i7;

      auto B_itr = B.lower_bound(B53);
      auto B_end = B.lower_bound(B_itr, B53+D8);

      if(B_itr == B_end) continue;

      auto G_itr = G.lower_bound(G67);

      for(; B_itr != B_end; ++B_itr)
      {
         size_t i8 = B_itr->first%D8;
         G_itr = G.get(G_itr, G67+i8);
         gemm(CblasNoTrans, CblasNoTrans, 1.0, Y_itr->second, B_itr->second, 1.0, G_itr->second);
      }
   }
}

/// renormalize operator: i.e. calculate G = W * A^dag * F * B (backward)
///
/// input tensors:
/// W [6,1,4,5]
/// F [1,2,3]
/// A [4,7,2]
/// B [5,8,3]
///
/// output tensor
/// G [6,7,8]
///
/// intermediate tensors
/// X [1,4,7,3] = F * A^dag
/// Y [6,5,7,3] = W * X
///
///  (7)---A---(2)---+
///        |         |
///       (4)        |
///        |         |
///  (6)---W---(1)---F
///        |         |
///       (5)        |
///        |         |
///  (8)---B---(3)---+
///
template<class Qn, typename T>
void __renormalize_bwd_sweep (
      const MPO<T, Qn, 1>& W,
      const MPS<T, Qn, 1>& A,
      const MPS<T, Qn, 1>& B,
      const BLOCK<T, Qn>& F,
            BLOCK<T, Qn>& G)
{
   // Explicit implementation of tensor contractions in renormalization

   const QnIndex<Qn>& Q1 = F.qn(0);
   const QnIndex<Qn>& Q3 = F.qn(2);
         QnIndex<Qn>  Q4 =-A.qn(0);
   const QnIndex<Qn>& Q5 = W.qn(3);
   const QnIndex<Qn>& Q6 = W.qn(1);
         QnIndex<Qn>  Q7 =-A.qn(2);
   const QnIndex<Qn>& Q8 = B.qn(2);

   BTAS_ASSERT(W.qn(0) == -Q1, "renormalize: mismatched quantum number index #1");
   BTAS_ASSERT(A.qn(1) == F.qn(1), "renormalize: mismatched quantum number index #2");
   BTAS_ASSERT(B.qn(1) == -Q3, "renormalize: mismatched quantum number index #3");
   BTAS_ASSERT(W.qn(2) == A.qn(0), "renormalize: mismatched quantum number index #4");
   BTAS_ASSERT(B.qn(0) == -Q5, "renormalize: mismatched quantum number index #5");

   // X[1,4,7,3] = F[1,2,3] * A[4,2,7]^dag

   QnTensor<Qn, SpTensor<DnTensor<T, 2>, 4, CblasRowMajor>> Y(F.q()-A.q(),Q1,Q4,Q7,Q3);

   auto F_itr = F.begin();
   auto X_itr = X.begin();

   for(size_t i1 = 0; i1 < D1; ++i1)
   {
      size_t F1 = i1*D23;
      size_t X1 = i1*D473;

           F_itr = F.lower_bound(F_itr, F1);
      auto F_end = F.lower_bound(F_itr, F1+D23);

      if(F_itr == F_end) continue;

      auto A_itr = A.begin();

      for(size_t i4 = 0; i4 < D4; ++i4)
      {
         size_t A4 = i4*D27;
         size_t X14 = X1+i4*D73;

              A_itr = A.lower_bound(A_itr, A4);
         auto A_end = A.lower_bound(A_itr, A4+D27);

         if(A_itr == A_end) continue;

         for(; A_itr != A_end; ++A_itr)
         {
            size_t i27 = A_itr->first%D27;
            size_t i2 = i27/D7;
            size_t i7 = i27%D7;

            size_t F12 = F1+i2*D3;
            size_t X147 = X14+i7*D3;

                 F_itr = F.lower_bound(F_itr, F12);
            auto F_jtr = F.lower_bound(F_itr, F12+D3);

            if(F_itr == F_end) continue;

            for(; F_itr != F_jtr; ++F_itr)
            {
               size_t i3 = F_itr->first%D3;

               X_itr = X.get(X_itr, X147+i3);

               gemm(CblasConjTrans, CblasNoTrans, 1.0, A_itr->second, F_itr->second, 1.0, X_itr->second);
            }
         }
      }
   }

   // Y[6,5,7,3] = W[1,6,4,5] * X[1,4,7,3]

   QnTensor<Qn, SpTensor<DnTensor<T, 2>, 4, CblasRowMajor>> Y(W.q()+X.q(),Q6,Q5,Q7,Q3);

   for(auto W_itr = W.begin(); W_itr != W.end(); ++W_itr)
   {
      size_t i14 = W_itr->first/D65;
      size_t i65 = W_itr->first%D65;

      size_t X14 = i14*D73;

      auto X_itr = X.lower_bound(X14);
      auto X_end = X.lower_bound(X_itr, X14+D73);

      if(X_itr == X_end) continue;

      size_t Y65 = i65*D73;

      auto Y_itr = Y.lower_bound(Y65);

      for(; X_itr != X_end; ++X_itr)
      {
         size_t i73 = X_itr->first%D73;

         Y_itr = Y.get(Y_itr, Y65+i73);

         axpy(W_itr->second, X_itr->second, Y_itr->second);
      }
   }

   // G[6,7,8] = Y[6,5,7,3] * B[5,3,8]

   if(G.size() > 0)
   {
      BTAS_ASSERT(G.qn(0) == Q6, "renormalize: mismatched quantum number index #6");
      BTAS_ASSERT(G.qn(1) == Q7, "renormalize: mismatched quantum number index #7");
      BTAS_ASSERT(G.qn(2) == Q8, "renormalize: mismatched quantum number index #8");
   }
   else
   {
      G.resize(Y.q()+B.q(),Q6,Q7,Q8);
   }

   for(auto Y_itr = Y.begin(); Y_itr != Y.end(); ++Y_itr)
   {
      size_t i3 = Y_itr->first%D3;
      size_t i657 = Y_itr->first/D3;
      size_t i7 = i657%D7;
      size_t i65 = i657/D7;
      size_t i5 = i65%D5;
      size_t i6 = i65/D5;

      size_t B53 = i5*D3+i3;
      size_t G67 = i6*D7+i7;

      auto B_itr = B.lower_bound(B53);
      auto B_end = B.lower_bound(B_itr, B53+D8);

      if(B_itr == B_end) continue;

      auto G_itr = G.lower_bound(G67);

      for(; B_itr != B_end; ++B_itr)
      {
         size_t i8 = B_itr->first%D8;
         G_itr = G.get(G_itr, G67+i8);
         gemm(CblasNoTrans, CblasNoTrans, 1.0, Y_itr->second, B_itr->second, 1.0, G_itr->second);
      }
   }
}

/// gateway: renormalize operator
template<class Qn, typename T>
void renormalize (
      const bool& forward,
      const MPO<T, Qn, 1>& W,
      const MPS<T, Qn, 1>& A,
      const MPS<T, Qn, 1>& B,
      const BLOCK<T, Qn>& F,
            BLOCK<T, Qn>& G)
{
   if(forward)
      __renormalize_fwd_sweep(W, A, B, F, G);
   else
      __renormalize_bwd_sweep(W, A, B, F, G);
}

//
//  Merged version
//

/// renormalize operator: i.e. calculate G = A^dag * F * B (forward)
///
/// input tensors:
/// F [1,2,3]
/// A [2,4]
/// B [3,5]
///
/// output tensor
/// G [1,4,5]
///
/// intermediate tensors
/// X [1,4,3] = F * A^dag
///
///  +===(2)===A---(4)
///  |          
///  |           
///  |          
///  F---(1)-------(1)
///  |          
///  |           
///  |          
///  +===(3)===B---(5)
///
template<class Qn, typename T>
void __renormalize_fwd_sweep (
      const MPS<T, Qn, 0>& A,
      const MPS<T, Qn, 0>& B,
      const BLOCK<T, Qn>& F,
            BLOCK<T, Qn>& G)
{
   // Explicit implementation of tensor contractions in renormalization

   const QnIndex<Qn>& Q1 = F.qn(0);
   const QnIndex<Qn>& Q2 = F.qn(1);
   const QnIndex<Qn>& Q3 = F.qn(2);
         QnIndex<Qn>  Q4 =-A.qn(0);
   const QnIndex<Qn>& Q5 = B.qn(2);

   BTAS_ASSERT(A.qn(1) == F.qn(1), "renormalize: mismatched quantum number index #2");
   BTAS_ASSERT(B.qn(1) == -Q3, "renormalize: mismatched quantum number index #3");

   size_t D1 = Q1.size();
   size_t D2 = Q2.size();
   size_t D3 = Q3.size();
   size_t D4 = Q4.size();
   size_t D5 = Q5.size();

   size_t D23 = D2*D3;
   size_t D43 = D4*D3;
   size_t D45 = D4*D5;

   // X[1,4,3] = A[2,4]^dag * F[1,2,3]

   BLOCK<T, Qn> X(F.q()-A.q(),Q1,Q4,Q3);

   for(auto F_itr = F.begin(); F_itr != F.end(); ++F_itr)
   {
      size_t i12 = F_itr->frist/D3;
      size_t i3 = F_itr->first%D3;
      size_t i1 = i12/D2;
      size_t i2 = i12%D2;

      size_t A2 = i2*D4;
      size_t X13 = i1*D43+i3;

      auto A_itr = A.lower_bound(A2);
      auto A_end = A.lower_bound(A_itr, A2+D4);

      if(A_itr == A_end) continue;

      auto X_itr = X.lower_bound(X13);

      for(; A_itr != A_end; ++A_itr)
      {
         size_t i4 = A_itr->first%D4;

         X_itr = X.get(X_itr, X13+i4*D3);

         gemm(CblasConjTrans, CblasNoTrans, 1.0, A_itr->second, F_itr->second, 1.0, X_itr->second);
      }
   }

   // G[1,4,5] = F[1,4,3] * B[3,5]

   if(G.size() > 0)
   {
      BTAS_ASSERT(G.qn(0) == Q1, "renormalize: mismatched quantum number index #1");
      BTAS_ASSERT(G.qn(1) == Q4, "renormalize: mismatched quantum number index #4");
      BTAS_ASSERT(G.qn(2) == Q5, "renormalize: mismatched quantum number index #5");
   }
   else
   {
      G.resize(X.q()+B.q(),Q1,Q4,Q5);
   }

   for(auto Y_itr = Y.begin(); Y_itr != Y.end(); ++Y_itr)
   {
      size_t i3 = Y_itr->first%D3;
      size_t i657 = Y_itr->first/D3;
      size_t i7 = i657%D7;
      size_t i65 = i657/D7;
      size_t i5 = i65%D5;
      size_t i6 = i65/D5;

      size_t B53 = i5*D3+i3;
      size_t G67 = i6*D7+i7;

      auto B_itr = B.lower_bound(B53);
      auto B_end = B.lower_bound(B_itr, B53+D8);

      if(B_itr == B_end) continue;

      auto G_itr = G.lower_bound(G67);

      for(; B_itr != B_end; ++B_itr)
      {
         size_t i8 = B_itr->first%D8;
         G_itr = G.get(G_itr, G67+i8);
         gemm(CblasNoTrans, CblasNoTrans, 1.0, Y_itr->second, B_itr->second, 1.0, G_itr->second);
      }
   }
}

/// renormalize from merged block and mps
template<class Qn, typename T>
void renormalize (
      const bool& forward,
      const MPS<T, Qn, 0>& A,
      const MPS<T, Qn, 0>& B,
      const BLOCK<T, Qn>& F,
            BLOCK<T, Qn>& G)
{
   if(forward)
      __renormalize_fwd_sweep(A, B, F, G);
   else
      __renormalize_bwd_sweep(A, B, F, G);
}

} // namespace btas

#endif // __BTAS_MPSXX_DRIVER_RENORMALIZE_H
