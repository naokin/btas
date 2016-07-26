

template<class Iter> struct __ConstIterator;

template<typename T> struct __ConstIterator<T*> { typedef const T* type; };

template<typename T> struct __ConstIterator<T*> { typedef const T* type; };

template<typename T, class Alloc> struct __ConstIterator<typename std::vector<T,Alloc>::iterator>
{ typedef typename std::vector<T,Alloc>::const_iterator type; };

template<typename T, class Alloc> struct __ConstIterator<typename std::vector<T,Alloc>::const_iterator>
{ typedef typename std::vector<T,Alloc>::const_iterator type; };

template<class Iter, size_t N>
struct __ConstIterator< TensorIterator<Iter,N> >
{
  typedef TensorIterator<typename __ConstIterator<Iter>::type,N> type;
};

template<class Iter, size_t N>
class TensorIterator {

  friend class TensorIterator<typename __ConstIterator<Iter>::type,N>;
};
