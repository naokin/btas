/// pre-declaration
template<typename T, class TensorA, class TensorB, class TensorC>
void contract (
      const T& alpha,
      const TensorA& A,
      const std::string& symblA,
      const TensorB& B,
      const std::string& symblB,
      const T& beta,
            TensorC& C,
      const std::string& symblC);

/// generic implementation
template<typename T, class TensorA, class TensorB, class TensorC>
struct __contract_impl
{
   static void call (
      const T& alpha,
      const TensorA& A,
      const std::string& symblA,
      const TensorB& B,
      const std::string& symblB,
      const T& beta,
            TensorC& C,
      const std::string& symblC)
   {
      // implementation goes here
      call (alpha, A, symblA, B, symblB, beta, C, symblC,
               std::bind(contract<T,
                  typename TensorA::value_type,
                  typename TensorB::value_type,
                  typename TensorC::value_type>,
               alpha, _1, symblA, _2, symblB, beta, _3, symblC);
   }

   static void call (
      const T& alpha,
      const TensorA& A,
      const std::string& symblA,
      const TensorB& B,
      const std::string& symblB,
      const T& beta,
            TensorC& C,
      const std::string& symblC,
            std::function<void(
               const typename TensorA::value_type&,
               const typename TensorB::value_type&,
               const typename TensorC::value_type&)> FnContr)
   {
      // alias to rank
      constexpr size_t NA = A.rank();
      constexpr size_t NB = B.rank();
      constexpr size_t NC = C.rank();

      // implementation goes here
      Index<NA> indexA(symblA);
      Index<NB> indexB(symblB);
      Index<NC> indexC(symblC);
   }
};

template<typename T, size_t NA, size_t NB, size_t NC, CBLAS_ORDER Order>
struct __contract_impl<T, DnTensor<T, NA, Order>, DnTensor<T, NB, Order>, DnTensor<T, NC, Order>>
{
   static void call (
      const T& alpha,
      const DnTensor<T, NA, Order>& A,
      const std::string& indexA,
      const DnTensor<T, NB, Order>& B,
      const std::string& indexB,
      const T& beta,
            DnTensor<T, NC, Order>& C,
      const std::string& indexC)
   {
      // specialization goes here
   }
};

template<typename T, class TensorA, class TensorB, class TensorC>
void contract (
      const T& alpha,
      const TensorA& A,
      const std::string& indexA,
      const TensorB& B,
      const std::string& indexB,
      const T& beta,
            TensorC& C,
      const std::string& indexC)
{
   __contract_impl<T, TensorA, TensorB, TensorC>::call(alpha, A, indexA, B, indexB, beta, C, indexC);
}


// C[ij,kl] = A[ij,pq] x B[pq,kl]

{
   for(auto A_itr = A.begin(); A_itr != A.end(); ++A_itr)
   {
      ij = A_itr->first/Dpq;
      pq = A_itr->first%Dpq;
      auto B_itr = B.lower_bound(pq); // cost: nnz_A * log_2 nnz_B
      auto B_end = B.lower_bound(B_itr, pq+Dkl);
      for(; B_itr != B_end; ++B_itr)
      {
         kl = B_itr->first%Dkl;
         gemm(A_itr->second, B_itr->second, C(ij*Dkl+kl));
      }
   }
}

{
   for(auto it = B.begin(); it != B.end(); ++it)
   {
   }
}
