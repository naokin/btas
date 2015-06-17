#include <iostream>

#include <boost/bind.hpp>

template<class Op>
void carry (int value, Op set) { set(value); }

class foo {

public:

  foo () : data_(0) { }

  void disp () const { std::cout << data_ << std::endl; }

  void set (int value)
  { carry(value,boost::bind(&foo::set_,boost::ref(*this),_1)); }

private:

  void set_ (int value) { data_ = value; }

  int data_;

};

int main ()
{
  foo a;

  a.disp();

  a.set(1);

  a.disp();

  return 0;
}
