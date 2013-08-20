#ifndef _BTAS_DAVIDSON_H
#define _BTAS_DAVIDSON_H 1

namespace btas
{

template<typename T>
void Davidson(const DavidsonFunctor<T>& functor,
                    std::vector<double>& eigen_values,
                    std::vector<T>& eigen_vectors,
              const DavidsonSetup)
{
}

};

#endif // _BTAS_DAVIDSON_H
