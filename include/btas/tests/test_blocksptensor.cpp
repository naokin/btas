#include <iostream>
#include <iomanip>

#include <boost/mpi.hpp>

#include <btas.h>
#include <btas/BlockSpTensor.hpp>

#include "fermion.h"

int main (int argc, char* argv[])
{
  using namespace btas;

  boost::mpi::environment env(argc,argv);
  boost::mpi::communicator world;

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(1);

  typedef BlockSpTensor<double,4,fermion> tensor_t;
  typedef tensor_t::qnum_array_type qarray_t;
  typedef tensor_t::qnum_shape_type qshape_t;
  typedef tensor_t::size_array_type narray_t;
  typedef tensor_t::size_shape_type nshape_t;

  qarray_t qa;
  qa.push_back(fermion(0, 0));
  qa.push_back(fermion(1, 1));
  qa.push_back(fermion(1,-1));
//qa.push_back(fermion(2, 2));
  qa.push_back(fermion(2, 0));
//qa.push_back(fermion(2,-2));
//qa.push_back(fermion(3, 3));
//qa.push_back(fermion(3, 1));
//qa.push_back(fermion(3,-1));
//qa.push_back(fermion(3,-3));
//qa.push_back(fermion(4, 4));
//qa.push_back(fermion(4, 2));
//qa.push_back(fermion(4, 0));
//qa.push_back(fermion(4,-2));
//qa.push_back(fermion(4,-4));
//qa.push_back(fermion(5, 5));
//qa.push_back(fermion(5, 3));
//qa.push_back(fermion(5, 1));
//qa.push_back(fermion(5,-1));
//qa.push_back(fermion(5,-3));
//qa.push_back(fermion(5,-5));

  qshape_t qs = make_array(qa,qa,conj(qa),conj(qa));

  narray_t na(qa.size(),100);
  nshape_t ns = make_array(na,na,na,na);

  tensor_t A(fermion(0,0),qs,ns);

  size_t iproc = world.rank();
  double value = 0.1;
  A.fill(value);

/*
  for(size_t i = 0; i < A.size(); ++i) {
    if(A.has(i)) {
      // send to proc# 0
      tensor_t::const_iterator it = A.get(i,0);
      if(world.rank() == 0) {
        const tensor_t::tile_type& a = *it;
        std::cout << std::setw(3) << i << " at p" << A.where(i) << " :: ";
        for(size_t j = 0; j < a.size(); ++j)
          std::cout << std::setw(4) << a[j];
        std::cout << std::endl;
      }
    }
  }
*/

  double dotA = dotc(A,A);
  size_t nnzA = A.nnz();
  if(world.rank() == 0) {
    std::cout << "nnz = " << nnzA << std::endl;
    std::cout << "    = " << dotA << std::endl;
  }

  return 0;
}
