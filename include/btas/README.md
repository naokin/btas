Environmental variables used in compilation:
----

_HAS_CBLAS
_HAS_INTEL_MKL
_SERIAL

Class definitions:
----

- Tensor, A tensor class implemented in terms of STL vector as strage. Compilable with GCC-4.4 or later using -std=c++0x flag.

- TensorWrapper, A class wrapping a pointer to the first of an array.

- TensorIterator, Iterator in terms of tensor index.

- TensorView, A class wrapping a TensorIterator to provide a generic tensor view object.

