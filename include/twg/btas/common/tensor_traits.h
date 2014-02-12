#ifndef __BTAS_TENSOR_TRAITS_H
#define __BTAS_TENSOR_TRAITS_H 1

#include <iterator>
#include <type_traits>

namespace btas {

/// test T has data() member
/// this will be used to detect whether or not the storage is consecutive
template<class T>
class has_data
{
   /// true case
   template<class U>
   static auto __test(U* p) -> decltype(p->data(), std::true_type());
   /// false case
   template<class>
   static std::false_type __test(...);
public:
   static constexpr const bool value = std::is_same<std::true_type, decltype(__test<T>(0))>::value;
};

/// test T has rank() member
template<class T>
class has_rank
{
   /// true case
   template<class U>
   static auto __test(U* p) -> decltype(p->rank(), std::true_type());
   /// false case
   template<class>
   static std::false_type __test(...);
public:
   static constexpr const bool value = std::is_same<std::true_type, decltype(__test<T>(0))>::value;
};

/// test T has value_type
template<class T>
class has_value_type
{
   /// true case
   template<class U>
   static std::true_type __test(typename U::value_type*);
   /// false case
   template<class>
   static std::false_type __test(...);
public:
   static constexpr const bool value = std::is_same<std::true_type, decltype(__test<T>(0))>::value;
};

/// get deepest data type
template<typename T, bool = has_value_type<T>::value>
struct element_type
{
   typedef T type;
};

/// get deepest data type
template<typename T>
struct element_type<T, true>
{
   typedef typename element_type<T>::type type;
};

} // namespace btas

#endif // __BTAS_TENSOR_TRAITS_H
